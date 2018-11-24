import torch
from torch import optim
from torch.utils import data

from util.data import DS as ds
from model.base import Inpainter
from loss.nvidia import CompositeLoss as loss_func

#      - ensure model is wired properly
#      - tensorboard logging
#      - model persistence
#      - mask dataset generation

def train(model, args):
    """ training loop
        Input
        ------
            model - the model architecture to train on
            args - hyperparams for our model

        set up our data loader, toggle the model to train, and roll through a few epochs
    """
    loader = data.DataLoader(ds(img='data/b1/img/', mask='data/b1/mask/'),
                             batch_size=2, num_workers=1, pin_memory=False)
    model.train()
    optimizer = optim.Adam([p for p in  model.parameters()], lr=args['learning_rate'])

    loss = loss_func()
    ctr = 0

    while True:
        for images, masks, emasks in loader:
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
    MODEL = Inpainter()
    MODEL.init_weights()
    ARGS = {'learning_rate' : 0.00005}

    train(MODEL, ARGS)
