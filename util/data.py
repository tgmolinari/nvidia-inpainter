import os
import torch.utils.data as data
from skimage import io

class DS(data.Dataset):
    def __init__(self, img, mask, transform = None):
        self.imgs = os.listdir(img)
        self.masks = os.listdir(masks)
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

        if transform is not None:
            img = self.transform(img)
        
        return img, mask
