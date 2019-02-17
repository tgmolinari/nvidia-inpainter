""" partial convolution implementation """
import torch
import torch.nn.functional as F

class PConv2d(torch.nn.Conv2d):# bias false per the karpathy tweet 6/30/18
    """A partial convolution layer, implemented from Liu et al (2018)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,\
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=0,
                         dilation=dilation, groups=groups, bias=bias)
        self.mask_filters = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        self.window_size = in_channels * kernel_size * kernel_size
        self.pad = padding

    def __call__(self, inp, mask):
        return self.pforward(inp, mask)

    def pforward(self, inp, mask):
        """
            The partial convolution forward operation
            Parameters:
            ------
            inp : the input tensor
            mask : the mask tensor, which is a map of 0's and 1's corresponding to the points
                        in our image tensor that have been convolved with unmaked pixels
            Returns
            ------
            f : the featuremap of a 2D partial convolution after the forward
            new_mask : updated mask to cover parts of the featuremap that have received a bit
                        of influence from the unobscured parts of the image
        """
        if self.pad:
            mask = F.pad(mask, (self.pad, self.pad, self.pad, self.pad), 'reflect')
            inp = F.pad(inp, (self.pad, self.pad, self.pad, self.pad), 'reflect')
        # update after seeing Liu's imp
        # The mask_ratio from his imp is the gamechanger, should solve the numerical instability (lots of v small numbers leading to nans)
        # much nicer idiom for skipping gradient calcs
        with torch.no_grad():
            # performs Equation 2
            new_mask = F.conv2d(mask, self.mask_filters,\
                                                stride=self.stride, padding=0)
            mask_ratio = self.window_size/(new_mask + 1e-8)            
            new_mask = torch.clamp(new_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, new_mask)
        # performs the operation in Equation 1
        f = self.forward((inp * mask))
        f = torch.mul(f, mask_ratio)

        return f, new_mask
