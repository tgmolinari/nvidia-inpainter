import torch
import torch.optim as optim
import torch.utils.data as data

from util.data import DS as ds
from model.base import inpainter
from loss.nvidia import CompositeLoss as loss_func
from util.junkdrawer import show

# TODO - ensure model is wired properly
#      - tensorboard logging
#      - model persistence
#      - mask dataset generation

def train(model, args):
    loader = data.DataLoader(ds(img = 'data/b1/img/', mask ='data/b1/mask/'),
        batch_size = 2, num_workers = 1, pin_memory = False)
    learning_rate = 0.00005
    model.train()
    optimizer = optim.Adam([p for p in  model.parameters()], lr = learning_rate)

    loss = loss_func()
    ctr = 0

    while True:
        for i, (images, masks, emasks) in enumerate(loader):

            preds = model.forward(images, masks)

            optimizer.zero_grad()
            bloss = loss(preds, images, masks, emasks)
            bloss.backward()
            optimizer.step()

            ctr += 1

            if ctr % 20 == 0:
                print('writing first ', str(ctr), ' to disk')
                torch.save(preds, 'curr_preds' + str(ctr) + '.pt')


if __name__ == '__main__':
    model = inpainter()
    model.init_weights()
    args = {}

    train(model, args)
