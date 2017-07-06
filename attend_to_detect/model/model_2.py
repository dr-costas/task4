#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import absolute_import
import sys

import torch
from torch.autograd import Variable
from torch.nn import functional

from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping, ScaleAndShift

from sklearn.preprocessing import StandardScaler

import numpy as np

if sys.version_info > (3, 0):
    from attend_to_detect.model.category_specific_branch import CategoryBranch
else:
    from .model.category_specific_branch import CategoryBranch

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


def padder(data):
    data = list(data)
    for index in [0, -2, -1]:
        max_ts = np.max([datum.shape[-2] for datum in data[index]])

        for i in range(len(data[index])):
            len_dif = max_ts - data[index][i].shape[-2]
            if len_dif > 0:
                data[index][i] = np.concatenate((
                    data[index][i],
                    np.zeros((1, len_dif, data[index][i].shape[-1]))),
                    axis=-2
                )
                if index != 0:
                    data[index][i][:, -len_dif:, 0] = 1
    data = tuple(data)

    return data


class CommonFeatureExtractor(torch.nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding,
                 dilation=1, activation=functional.leaky_relu):
        super(CommonFeatureExtractor, self).__init__()
        self.conv = torch.nn.Conv2d(
            1, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation

    def forward(self, x):
        return self.bn(self.activation(self.conv(x)))


def get_data_stream(batch_size, dataset_name='dcase_2017_task_4_test.hdf5',
                    that_set='train', scale=True, use_scaler=True, scaler_mean=None,
                    scaler_std=None):
    dataset = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)

    scaler = StandardScaler()

    stream = DataStream(dataset=dataset, iteration_scheme=scheme)

    mean_val = scaler_mean
    std_val = scaler_std

    if scale:
        for data in stream.get_epoch_iterator():
            for example in data[0]:
                scaler.partial_fit(example)

        mean_val = scaler.mean_
        std_val = scaler.std_
    if use_scaler:
        if all([i is None for i in [mean_val, std_val]]):
            raise AttributeError('Provide mean and std values for scaler')
        stream = ScaleAndShift(
            data_stream=stream,
            scale=-mean_val/std_val,
            shift=1/std_val,
            sources=('audio_features', )
        )

    stream = Mapping(stream, mapping=padder)

    return stream, mean_val, std_val


def main():
    batch_size = 64
    nb_filters = 40
    kernel_size = 3

    epochs = 100
    patience = 15

    common_feature_extractor = CommonFeatureExtractor(
        out_channels=nb_filters, kernel_size=kernel_size, stride=1, padding=1
    )

    branch_alarm = CategoryBranch()
    branch_vehicle = CategoryBranch()

    loss_fn_alarm = torch.nn.CrossEntropyLoss()
    loss_fn_vehicle = torch.nn.CrossEntropyLoss()

    optim_common = torch.optim.Adam()
    optim_alarm = torch.optim.Adam()
    optim_vehicle = torch.optim.Adam()

    train_data, mean_val, std_val = get_data_stream(
        batch_size=batch_size
    )

    valid_data, _, __ = get_data_stream(
        batch_size=batch_size, that_set='test', scale=False,
        scaler_mean=mean_val, scaler_std=std_val
    )

    not_improved_counter = 0
    prv_loss = None
    current_loss = None

    for epoch in range(epochs):

        for batch in train_data.get_epoch_iterator():
            common_features = common_feature_extractor(batch[0])
            alarm_output = branch_alarm(common_features)
            vehicle_output = branch_vehicle(common_features)

            common_feature_extractor.zero_grad()
            branch_alarm.zero_grad()
            branch_vehicle.zero_grad()

            loss_alarm = loss_fn_alarm()
            loss_vehicle = loss_fn_vehicle()

            loss_alarm.backward(retain_variables=True)
            optim_alarm.step()
            optim_common.step()

            common_feature_extractor.zero_grad()

            loss_vehicle.backward()
            optim_vehicle.step()
            optim_common.step()

        for batch in valid_data.get_epoch_iterator():
            common_features = common_feature_extractor(batch[0])
            alarm_output = branch_alarm(common_features)
            vehicle_output = branch_vehicle(common_features)

            loss_alarm = loss_fn_alarm()
            loss_vehicle = loss_fn_vehicle()


if __name__ == '__main__':
    main()

# EOF
