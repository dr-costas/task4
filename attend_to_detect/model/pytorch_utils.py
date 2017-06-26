import torch
from torch.autograd import Variable

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

    def __init__(self, model,
            optimizer, criterion,
            train_fn, train_iter,
            valid_fn, valid_iter,
            checkpoint_path=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_fn = train_fn
        self.train_iter = train_iter
        self.valid_fn = valid_fn
        self.valid_iter = valid_iter
        if checkpoint_path is None:
            # Save to current directory
            self.checkpoint_path = os.getcwd()
        else:
            self.checkpoint_path = checkpoint_path
        self.history = {'train_loss': [],
                'valid_loss': []}
        self.total_iters = 0
        self.cur_epoch = 0

    def train(self, n_epochs=1, n_workers=1, save_every=None):
        # Note: Logging expects the losses to be divided by the batch size

        # Set up
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        save_every_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}it.pt')
        save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')

        for epoch in range(self.cur_epoch, n_epochs):
            print('Epoch {}/{}'.format(epoch+1, n_epochs))
            train_iter = tqdm(enumerate(self.train_iter))
            self.history['train_loss'].append([])
            train_loss = self.history['train_loss'][-1]
            for t, batch in train_iter:
                train_loss.append(self.train_fn(self.model,
                    self.optimizer,
                    self.criterion,
                    batch))
                train_iter.set_postfix(loss=np.mean(train_loss))
                self.total_iters += 1
                if save_every is not None:
                    if self.total_iters % save_every == 0:
                        torch.save(self, save_every_fmt.format(self.total_iters))

            # Validation
            val_loss = 0.0
            for t, batch in enumerate(self.valid_iter):
                val_loss += self.valid_fn(self.model, self.criterion, batch)
            val_loss /= t+1
            print('Validation loss: {}'.format(val_loss))
            self.history['valid_loss'].append(val_loss)

            self.cur_epoch += 1

            # Checkpointing
            print('Checkpointing...')
            ckpt = {'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'history': self.history,
                    'total_iters': self.total_iters,
                    'cur_epoch': self.cur_epoch}
            torch.save(ckpt, save_epoch_fmt.format(epoch))


    def load_checkpoint(self, ckpt):
        ckpt = torch.load(ckpt)
        # Load model state
        self.model.load_state_dict(ckpt['model_state'])
        # Load optimizer state
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        # Load history
        self.history = ckpt['history']
        self.total_iters = ckpt['total_iters']
        self.cur_epoch = ckpt['cur_epoch']


if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader
    # Setup dummy model and optimizer
    model = torch.nn.Sequential(torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10))
    opt = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    train_data = TensorDataset(torch.rand(64, 10),
            torch.rand(64, 10))
    valid_data = TensorDataset(torch.rand(32, 10),
            torch.rand(32, 10))

    train_iter = DataLoader(train_data, 8, shuffle=True)
    valid_iter = DataLoader(valid_data, 8)

    def train_fn(model, optimizer, criterion, batch):
        x, y = batch
        x = Variable(x)
        y = Variable(y, requires_grad=False)

        y_hat = model.forward(x)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.data[0]

    def valid_fn(model, criterion, batch):
        x, y = batch
        x = Variable(x)
        y = Variable(y, requires_grad=False)

        y_hat = model.forward(x)
        loss = criterion(y_hat, y)

        return loss.data[0]

    print('Testing creation of TrainLoop')
    tl = TrainLoop(model, opt, criterion,
            train_fn, train_iter,
            valid_fn, valid_iter,
            checkpoint_path='test_ckpt')

    print('Testing tl.train')
    tl.train(n_epochs=5)

    print('Testing tl.load_checkpoint')
    tl.load_checkpoint('test_ckpt/checkpoint_4ep.pt')

    print('Testing resuming from checkpoint')
    tl.train(n_epochs=10)

