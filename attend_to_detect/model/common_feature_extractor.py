#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import torch
from torch.nn import functional
from torch.nn.init import xavier_normal, constant

__docformat__ = 'reStructuredText'


class CommonFeatureExtractor(torch.nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding,
                 dropout, dilation=1, activation=functional.leaky_relu,
                 max_pool_kernel=None, max_pool_stride=None, max_pool_padding=None):
        super(CommonFeatureExtractor, self).__init__()
        self.conv = torch.nn.Conv2d(
            1, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation
        self.dropout = torch.nn.Dropout2d(dropout)

        if all([i is not None for i in [max_pool_kernel, max_pool_stride, max_pool_padding]]):
            self.max_pool_kernel = max_pool_kernel
            self.max_pool_stride = max_pool_stride
            self.max_pool_padding = max_pool_padding
            self.max_pool_layer = torch.nn.MaxPool2d(
                kernel_size=self.max_pool_kernel,
                stride=self.max_pool_stride,
                padding=max_pool_padding
            )
        else:
            self.max_pool_layer = None

        self.initialize()

    def initialize(self):
        xavier_normal(self.conv.weight.data)
        constant(self.conv.bias.data, 0)

    def forward(self, x):
        to_return = self.bn(self.activation(self.conv(self.dropout(x))))
        if self.max_pool_layer is not None:
            return self.max_pool_layer(to_return)
        return to_return


def main():
    pass


if __name__ == '__main__':
    main()

# EOF
