#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable

from .model.category_specific_branch import CategoryBranch

class AttendToDetect(nn.Module):

    def __init__(self, input_dim, decoder_dim, output_classes,
            common_filters=32, common_stride=(2,2), common_kernel_size=(3,3),
            enc_filters=64, enc_stride=(2, 2), enc_kernel_size=(3,3),
            monotonic_attention=False, bias=False):
        super(AttendToDetect, self).__init__()
        self.common = nn.Conv2D(1, common_filters,
                kernel_size=common_kernel_size,
                stride=common_stride,
                padding=((common_kernel_size[0]-1)//2, (common_kernel_size[1]-1)//2)
                )
        self.categoryA = CategoryBranch()
        self.categoryB = CategoryBranch()


    def forward(self, x, n_steps):
        common_feats = self.common(x)
        predA, weightsA = self.categoryA(common_feats, n_steps)
        predB, weightsB = self.categoryB(common_feats, n_steps)

        return (predA, weightsA), (predB, weightsB)


def train_fn(model, optimizer, criterion, batch):
    x, y = batch

    x = Variable(torch.from_numpy(x.astype('float32')).cuda())
    y = Variable(torch.from_numpy(y.astype('float32')).cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    y_hat = model.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0]


def valid_fn(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda(), volatile=True)
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    hidden = model.init_hidden(x.size(0))
    y_hat = model.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    val_loss = criterion(y_hat, y).data[0]
    return val_loss


if __name__ == '__main__':
    x_ = Variable(torch.randn(5, 4, 161).cuda())
    y = Variable(torch.randn(5, 4, 161).cuda(), requires_grad=False)
    model = RNNModelWithSkipConnections(161, 20, 20, 161).cuda()
    h0_ = model.init_hidden(4)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for k in range(1000):
        y_hat = model.forward(x_, h0_)
        loss = criterion(y_hat, y)
        print('It. {}: {}'.format(k, loss.cpu().data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

