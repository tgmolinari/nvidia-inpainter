import torch
import torch.optim as optim
import torch.utils.data as data

from util.data import DS as ds
from model.base import inpainter
from loss.nvidia import CompositeLoss as loss_func
from util.junkdrawer import show
'''Initial thoughts re: naming convention and folder structure, gonna need to get a data loader working
    this'll be the __main__ for all of our effort!!!!!!!!!!!!!!!!!!!!!!!!
'''

def train(model, args):
    loader = data.DataLoader(ds(img = 'data/b1/img/', mask ='data/b1/mask/'),
        batch_size = 6, num_workers = 2, pin_memory = True)
    learning_rate = 0.00005
    model.train()
    optimizer = optim.Adam([p for p in  model.parameters()], lr = learning_rate)

    loss = loss_func()
    while True:
        ctr = 0
        for i, (images, masks) in enumerate(loader):
            preds = model.forward(images, masks)
            
            optimizer.zero_grad()
            bloss = loss(preds, images, masks)
            bloss.backward()
            optimizer.step()
            
            ctr += 1
            
            if ctr % 100 == 0:
                torch.save(preds, 'curr_preds' + str(ctr) + '.pt')


if __name__ == '__main__':
    model = inpainter()
    model.init_weights()
    args = {}

    train(model, args)
