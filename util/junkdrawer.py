import numpy as np
import torch
import matplotlib.pyplot as plt

def show(img):
    '''print a 3D tensor or numpy array, taken from a Soumith post somewhere on the pytorch forums'''
    if type(img) == torch.Tensor: # if we want to not import torch, can do https://stackoverflow.com/questions/49577290/determine-if-object-is-of-type-foo-without-importing-type-foo
        img = img.int()
        if img.dim() == 4:
            npimg = img.squeeze(0).numpy()
        else:
            npimg = img.numpy()

        img = npimg.transpose((1,2,0))

    plt.imshow(img, interpolation='nearest')
