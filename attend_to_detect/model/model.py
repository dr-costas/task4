#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

if sys.version_info > (3, 0):
    from attend_to_detect.model.category_specific_branch import CategoryBranch
else:
    from .model.category_specific_branch import CategoryBranch


# def padder(data, indices):
#     data = list(data)
#     masks = []
#     for index in indices:
#         index_masks = []
#         max_ts = np.max([datum.shape[-2] for datum in data[index]])
#
#         for i in range(len(data[index])):
#             len_dif = max_ts - data[index][i].shape[-2]
#             tmp = np.ones(data[index][i].shape[:-2] + (max_ts, ) + data[index][i].shape[-1:])
#             if len_dif > 0:
#                 data[index][i] = np.concatenate((
#                     data[index][i],
#                     np.zeros((1, len_dif, data[index][i].shape[-1]))),
#                     axis=-2
#                 )
#                 if index != 0:
#                     data[index][i][:, -len_dif:, 0] = 1
#
#     data = tuple(data)
#
#     return data


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


def train_fn(layers, optim, loss_criterion, batch):

    padded_batch = padder(batch)

    x, y = batch

    x = Variable(torch.from_numpy(x.astype('float32')).cuda())
    y = Variable(torch.from_numpy(y.astype('float32')).cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    y_hat = layers.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    loss = loss_criterion(y_hat, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

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


def main():
    x_ = Variable(torch.randn(5, 4, 161).cuda())
    y = Variable(torch.randn(5, 4, 161).cuda(), requires_grad=False)
    model = RNNModelWithSkipConnections(161, 20, 20, 161).cuda()
    h0_ = model.init_hidden(4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for k in range(1000):
        y_hat = model.forward(x_, h0_)
        loss = criterion(y_hat, y)
        print('It. {}: {}'.format(k, loss.cpu().data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

