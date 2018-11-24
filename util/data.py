""" dataloader and manipulation class """
import os
import numpy as np
import torch
from torch.utils import data
from skimage import io, transform as imtransform

def _expand(tensor):
    """ Function that expands the dilation of the mask tensor by one pixel in every direction
        The breakdown is
            - Get the width and height of the overall image

            - create our new mask tensor

            - set the tensor to be iterated over to have just one channel. This works bc we're guaranteed a nonzero val across all channels for a masked pixel.
            it simplifies pixelwise comparison in our tensor by escaping the ambiguity of boolean evaluation of multiple values

            - for each row i, for each column j:
                set the pixel value to the current index were looking at
                if we're not at the last or first column
                    if we're in the first row, look below and on either side
                    if we're in the last row, look above and on either side
                    otherwise, look above, below, and on either side
                if we're in the first column
                    same as above conditions, except don't look at the previous column
                if we're in the last column
                    same as first set of conditions, except don't look at next column
                set the pixel value to be the result of all these comparisons
            return our tensor!
    """
    xdim, ydim = tensor.size()[-2:]
    new_tensor = torch.zeros(tensor.size())
    tensor = tensor[0, :, :]
    for i in range(xdim):
        for j in range(ydim):
            val = tensor[i, j]
            if j != ydim - 1 and j != 0:
                if i == 0:
                    val = tensor[i+1, j] or tensor[i, j-1] or tensor[i, j+1] or val

                elif i == xdim - 1:
                    val = tensor[i-1, j] or tensor[i, j+1] or tensor[i, j-1] or val

                else:
                    val = tensor[i+1, j] or tensor[i-1, j] or tensor[i, j+1] or tensor[i, j-1] or val

            elif j == 0:
                if i == 0:
                    val = tensor[i+1, j] or tensor[i, j+1] or val
                elif i == xdim - 1:
                    val = tensor[i-1, j] or tensor[i, j+1] or val
                else:
                    val = tensor[i+1, j] or tensor[i-1, j] or tensor[i, j+1] or val

            elif j == ydim - 1:
                if i == 0:
                    val = tensor[i+1, j] or tensor[i, j-1] or val
                elif i == xdim - 1:
                    val = tensor[i-1, j] or tensor[i, j-1] or val
                else:
                    val = tensor[i+1, j] or tensor[i-1, j] or tensor[i, j-1] or val

            new_tensor[:, i, j] = val

    return new_tensor


class DS(data.Dataset):
    """ basic pytorch Dataset, need to override:
            __init__
            __len__
            __getitem__
    """
    def __init__(self, img, mask, transform=None):
        self.imgs = [img + i for i in os.listdir(img)]
        self.masks = [mask + m for m in os.listdir(mask)]
        self.mlen = len(self.masks)
        self.transform = transform


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img_path = self.imgs[index]
        mask_path = self.masks[np.random.randint(0, self.mlen)]
        img = io.imread(img_path)
        if img.shape[0] != img.shape[1] and img.shape[0] != 512:
            img = imtransform.resize(img, (512, 512), mode='constant', anti_aliasing=True)
        mask = io.imread(mask_path)
        img = img / 255

        if self.transform is not None:
            img = self.transform(img)
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        emask = 1 - _expand(mask)
        mask = 1 - mask

        return img, mask, emask
