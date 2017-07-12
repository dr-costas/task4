#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import torch
from torch.nn import functional
from torch.nn.init import xavier_normal, constant

__author__ = 'Konstantinos Drossos - TUT | Joao Santos - MILA | Dmitriy Serdyuk - MILA'
__docformat__ = 'reStructuredText'


class CommonFeatureExtractor(torch.nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding,
                 dropout, dilation=1, activation=functional.leaky_relu):
        super(CommonFeatureExtractor, self).__init__()
        self.conv = torch.nn.Conv2d(
            1, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation
        self.dropout = torch.nn.Dropout2d(dropout)

        self.initialize()

    def initialize(self):
        xavier_normal(self.conv.weight.data)
        constant(self.conv.bias.data, 0)

    def forward(self, x):
        return self.bn(self.activation(self.conv(self.dropout(x))))


def main():
    pass


if __name__ == '__main__':
    main()

# EOF
