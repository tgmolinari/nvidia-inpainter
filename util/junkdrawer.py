""" collection of utility code, inspired by https://twitter.com/betsythemuffin/status/1003313844108824584 """
import torch
import matplotlib.pyplot as plt

def show(img):
    """ print a 3D tensor or numpy array, modified mildly from this Soumith post:
    https://discuss.pytorch.org/t/how-to-show-a-image-in-jupyter-notebook-with-pytorch-easily/1229/4 
    """
    # if we want to not import torch, can do https://stackoverflow.com/questions/49577290/determine-if-object-is-of-type-foo-without-importing-type-foo
    if isinstance(img, torch.Tensor):
        if img.requires_grad:
            img = img.detach()
        if img.dim() == 4:
            npimg = img.squeeze(0).numpy()
        else:
            npimg = img.numpy()

        img = npimg.transpose((1, 2, 0))

    plt.imshow(img, interpolation='nearest')
