import torch

def _expand(tensor):
    # gross helper function to get the 1 pixel dilation of the hole for calculating total variation loss
    assert len(tensor.size()) == 4, "expected 4D tensor, got tensor of size %d" % len(tensor.size())
    new_tensor = torch.zeros(tensor.size())
    x, y = tensor.size()[-2:]
    for i in range(x):
        for j in range(y):
            val = 0
            if j != y - 1 and j != 0:
                if i == 0:
                    val = tensor[:,:,i+1, j] or tensor[:,:,i,j-1] or tensor[:,:,i,j+1] or tensor[:,:,i,j]

                elif i == x - 1:
                    val =  tensor[:,:,i-1,j] or tensor[:,:,i,j+1] or tensor[:,:,i,j-1] or tensor[:,:,i,j]

                else:
                    val = tensor[:,:,i+1, j] or tensor[:,:,i-1,j] or tensor[:,:,i,j+1] or tensor[:,:,i,j-1] or tensor[:,:,i,j]

            elif j == 0: 
                if i == j:
                    val = tensor[:,:,i+1, j] or tensor[:,:,i,j+1] or tensor[:,:,i,j]
                elif i == x - 1:
                    val =  tensor[:,:,i-1,j] or tensor[:,:,i,j+1] or tensor[:,:,i,j]
                else:
                    val = tensor[:,:,i+1, j] or tensor[:,:,i-1,j] or tensor[:,:,i,j+1] or tensor[:,:,i,j]

            elif j == y - 1: 
                if i == j:
                    val =  tensor[:,:,i-1,j] or tensor[:,:,i,j-1] or tensor[:,:,i,j]
                elif i == 0:
                    val = tensor[:,:,i+1, j] or tensor[:,:,i,j-1] or tensor[:,:,i,j]
                else:
                    val = tensor[:,:,i+1, j] or tensor[:,:,i-1,j] or tensor[:,:,i,j-1] or tensor[:,:,i,j]

            new_tensor[:,:,i,j] = val

    return new_tensor





def hole(out, gt, mask):
    '''per-pixel loss for the obscured region'''
    return torch.abs((1 - mask) * (out - gt))

def valid(out, gt, mask):
    '''per-pixel loss for the unobscured region'''
    return torch.abs(mask * (out - gt))

def perceptual(out, comp, gt):
    '''perceptual loss, as defined by Gatys et al. (2015)'''
    # following Gatys convention, not strictly obvious to me which is content and which is style
    # check that these are only passing the tensor values and not the reference, if it's the ref then we're goofing subsequent loss calcs
    curr_out = out
    curr_comp = comp
    curr_gt = gt
    style = 0
    content = 0
    for psi in psis:
        curr_out = psi(curr_out)
        curr_gt = psi(curr_gt)
        curr_comp = psi(curr_comp)
        content += torch.abs(curr_out - curr_gt)
        style += torch.abs(curr_comp - curr_gt)
    return style + content

# Kn is 1/C_n*H_n*K_n (specifying K_n??)
# and apparently Kn is just a scalar based off the dim size of the matrix????????????????
# y thooooooooooooo
# tensor = (batch, channels, w, h)
# tensor.transpose(1,-1)
# probably want flags for batched images vs not batched
def style_out(psis, out, gt, batched = True):
    '''Gram auto-correlative style loss, from Johnson et al. (2016) for the raw output image'''
    curr_out = out
    curr_gt = gt
    style_outl = 0

    for psi in psis:
        curr_out = psi(curr_out)
        curr_gt = psi(curr_gt)
        if batched:
            sizes = curr_out.size()[1:]
        else:
            sizes = curr_out.size()
        curr_kn = 1
        for x in sizes:
            curr_kn *= x
        curr_kn = 1/curr_kn

        out_gram = curr_out.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])) @
                            curr_out.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])).transpose(1,-1)
        gt_gram = curr_gt.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])) @
                            curr_gt.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])).transpose(1,-1)
        style_outl += torch.abs(curr_kn * (out_gram - gt_gram))
    return style_outl

def style_comp(psis, comp, gt, batched = True):
    '''Gram auto-correlative style loss, from Johnson et al. (2016) for the composited output image'''
    curr_comp = out
    curr_gt = gt
    style_compl = 0

    for psi in psis:
        curr_comp = psi(curr_comp)
        curr_gt = psi(curr_gt)
        if batched:
            sizes = curr_comp.size()[1:]
        else:
            sizes = curr_comp.size()
        curr_kn = 1
        for x in sizes:
            curr_kn *= x
        curr_kn = 1/curr_kn

        comp_gram = curr_comp.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])) @
                            curr_comp.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])).transpose(1,-1)
        gt_gram = curr_gt.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])) @
                            curr_gt.view_as(torch.rand(1,sizes[0],sizes[1]*sizes[1])).transpose(1,-1)
        style_compl += torch.abs(curr_kn * (comp_gram - gt_gram))
    return style_compl

def tv(comp, mask):
    ''' The total variation loss, calculated over the obscured region dilated by an additional pixel '''
    # for each pixel in MASK (invert the mask so the blotted regions are 1s and the unmasked are 0)
    # grab the 1 pixel dilation from composite image
    inverted_mask = 1 - mask
    expanded_mask = expand(inverted_mask)
    diff_i = torch.zeros(comp.size())
    diff_j = torch.zeros(comp.size())
    diff_j[:, :, :, :-1] = comp[:, :, :, 1:]
    diff_i[:, :, :-1, :] = comp[:, :, 1:, :]
    diff_i = torch.abs(diff_i - comp)
    diff_j = torch.abs(diff_j - comp)
    diff_i = diff_i * expanded_mask
    diff_j = diff_j * expanded_mask
    tvl = torch.sum(diff_j + diff_i)

    return tvl
