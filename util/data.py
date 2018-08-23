import os
import torch
import numpy as np
import torch.utils.data as data
from skimage import io

def _expand(tensor):
    ''' Function that expands the dilation of the mask tensor by one pixel in every direction
        The breakdown is
            - Get the width and height of the overall image
            
            - create our new mask tensor
            
            - set the tensor to be iterated over to have just one channel. This works bc we're guaranteed a nonzero val across all channels for a masked pixel. 
            it simplifies pixelwise comparison of values in our tensor by escaping the ambiguity of boolean evaluation of multiple values

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
    '''
    x, y = tensor.size()[-2:]
    new_tensor = torch.zeros(tensor.size())
    tensor = tensor[0, :,:]
    for i in range(x):
        for j in range(y):
            val = tensor[i, j]
            if j != y - 1 and j != 0:
                if i == 0:
                    val = tensor[ i+1, j] or tensor[i, j-1] or tensor[i, j+1] or val

                elif i == x - 1:
                    val =  tensor[i-1, j] or tensor[i, j+1] or tensor[i, j-1] or val

                else:
                    val = tensor[i+1, j] or tensor[i-1, j] or tensor[i, j+1] or tensor[i, j-1] or val

            elif j == 0: 
                if i == 0:
                    val = tensor[i+1, j] or tensor[i, j+1] or val
                elif i == x - 1:
                    val =  tensor[i-1, j] or tensor[i, j+1] or val
                else:
                    val = tensor[i+1, j] or tensor[i-1, j] or tensor[i, j+1] or val

            elif j == y - 1: 
                if i == 0:
                    val = tensor[i+1, j] or tensor[i, j-1] or val
                elif i == x - 1:
                    val =  tensor[i-1, j] or tensor[i, j-1] or val
                else:
                    val = tensor[i+1, j] or tensor[i-1, j] or tensor[i, j-1] or val

            new_tensor[:, i, j] = val

    return new_tensor


class DS(data.Dataset):
    def __init__(self, img, mask, transform = None):
        self.imgs = [img + i for i in os.listdir(img)]
        self.masks = [mask + m for m in os.listdir(mask)]
        self.mlen = len(self.masks)
        self.transform = transform


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        # ultimately want to do the img preproc here, but will return an image and a mask together
        img_path = self.imgs[index]
        mask_path = self.masks[np.random.randint(0,self.mlen)]
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        img = img / 255

        if self.transform is not None:
            img = self.transform(img)
        img = img.transpose((2,0,1))
        mask = mask.transpose((2,0,1))
        img = torch.from_numpy(img).type(torch.FloatTensor) 
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        emask = _expand(mask)
        return img, mask, emask
