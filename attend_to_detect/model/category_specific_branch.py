#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import division
import torch

class CategoryBranch(torch.nn.Module):

    def __init__(self, n_bins, n_filters, kernel_shape=(3,3), stride=1,
            n_timesteps=10, rnn_dims=[256, 256], rnn_bias=True):
        super(CategoryBranch, self).__init__()

        self.conv = nn.Conv2d(
            1, n_filters,
            kernel_size=conv_shape,
            stride=stride,
            padding=(0, (kernel_shape-1)//2)
        )

        self.rnn_in_size = n_filters * floor((n_bins - kernel_shape[0] + 1)/2 + 1)

        self.rnn0 = nn.GRUCell(self.rnn_in_size, rnn_dims[0], bias=rnn_bias)
        self.rnn1 = nn.GRUCell(rnn_dims[0], rnn_dims[1], bias=rnn_bias)
        self.rnn2 = nn.GRUCell(rnn_dims[1], rnn_dims[2], bias=rnn_bias)
        self.rnns = [self.rnn0, self.rnn1, self.rnn2]

        self.out = None

        self.init_weights()

        self.n_bins = n_bins
        self.n_filters = n_filters
        self.rnn_dims = rnn_dims
        self.n_timesteps = n_timesteps


    def init_weights(self):
        pass


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = [Variable(weight.new(batch_size, n_hidden).zero_().cuda())
                  for n_hidden in self.rnn_dims]
        return hidden


    def forward(self, x, h0):
        # input shape: (batch_size, 1, n_frames, n_bins)
        # output_shape: (n_timesteps, batch_size, self.rnn_dims[-1])

        # FIXME Should we run the convolution across the entire sequence at once
        # (840/860 frames) or at each timestep (84/86 frames each) separately?
        y_conv = self.conv(x).view(
            x.size(0), self.rnn_in_size, x.size(3)
        )
        x_rnn = y_conv.transpose(1, 2).transpose(0, 1).contiguous()
        h1, h2, h3 = h0
        out = Variable(torch.zeros(self.n_timesteps, self.batch_size, self.rnn_dims[-1]).cuda())

        # FIXME Only needed if we **really** need to change dimensions in between
        # layers, otherwise torch.nn.GRU will do the job and is much faster.
        for t in range(self.n_timesteps):
            h1 = self.rnn0(x_rnn[t], h1)
            h2 = self.rnn1(h1, h2)
            h3 = self.rnn2(h2, h3)
            out[t] = h3
        return out

def main():
    pass


if __name__ == '__main__':
    main()

# EOF
