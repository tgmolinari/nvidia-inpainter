import numpy as np
import matplotlib.pyplot as plt

def show(img):
    '''print a 3D tensor, taken from a Soumith post somewhere on the pytorch forums'''
    img = img.int()
    if img.dim() == 4:
        npimg = img.squeeze(0).numpy()
    else:
        npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
