import torch

class NaivePConv2d(torch.nn.Conv2d):# bias false per the karpathy tweet 6/30/18
    '''A partial convolution layer, implemented from Liu et al (2018)'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, dilation = 1, groups=1, bias=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
        self.mask_filters = torch.ones(out_channels, in_channels, kernel_size, kernel_size)

    def __call__(self, inp, mask):
        return self.pforward(inp,mask)
        
    def pforward(self, inp, mask):
        '''
            The partial convolution forward operation
            Parameters:
                inp : the input "image" tensor, representing the current feature map corresponding to the pictuer
                mask : the mask tensor, which is a map of 0's and 1's corresponding to the points in our image tensor that have been 
                        convolved with unmaked pixels
            Returns:
                f : the featuremap of a 2D partial convolution after the forward
                new_mask : updated mask to cover parts of the featuremap that have received a bit of influence from the unobscured parts of 
                            the image
        '''
        # TODO : Implement pconv style padding 
        assert mask.requires_grad == False
        # performs Equation 2
        new_mask = torch.nn.functional.conv2d(mask, self.mask_filters, stride = self.stride, padding = self.padding)
        # performs the operation in Equation 1
        f = self.forward((inp*mask))/new_mask
        # after the operation described in Section 3.2 - Implementation, need to set values of sum(M) to 1
        new_mask[new_mask > 0] = 1
        return f, new_mask
