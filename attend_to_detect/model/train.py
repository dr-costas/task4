#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import absolute_import
import sys
import os
import pickle

from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping

import torch
from torch.autograd import Variable
from torch.nn import functional

from sklearn.preprocessing import StandardScaler
import numpy as np

if sys.version_info < (3, 0):
    from .category_branch_2 import CategoryBranch2
    from .common_feature_extarctor import CommonFeatureExtractor
else:
    from attend_to_detect.model.category_branch_2 import CategoryBranch2
    from attend_to_detect.model.common_feature_extractor import CommonFeatureExtractor

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


alarm_classes = [
    'Train horn',
    'Air horn',
    'Car alarm',
    'Reversing beeps',
    'Ambulance (siren)',
    'Police car (siren)',
    'fire truck (siren)',
    'Civil defense siren',
    'Screaming'
]

vehicle_classes = [
    'Bicycle',
    'Skateboard',
    'Car',
    'Car passing by',
    'Bus',
    'Truck',
    'Motorcycle',
    'Train'
]

def category_cost(out_hidden, target):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
    target_flat = target.view(-1)
    return torch.nn.functional.cross_entropy(out_hidden_flat, target_flat)


def total_cost(hiddens, targets):
    return category_cost(hiddens[0], targets[0]) + \
            category_cost(hiddens[1], targets[1])


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


def get_data_stream(batch_size, dataset_name='dcase_2017_task_4_test.hdf5',
                    that_set='train', calculate_scaling_metrics=True, old_dataset=True):
    dataset = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)

    scaler = StandardScaler()

    stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    stream = Mapping(stream, mapping=padder)

    if calculate_scaling_metrics:
        for data in stream.get_epoch_iterator():
            for example in data[0]:
                if old_dataset:
                    scaler.partial_fit(example.reshape(example.shape[1:]))

        return stream, scaler
    return stream, None


def get_input(data, scaler, old_dataset=True, volatile=False):
    if old_dataset:
        for i in range(len(data)):
            data[i] = data[i].reshape(data[i].shape[1:])
    x = np.zeros((data.shape[0], ) + data[0].shape)
    for i, datum in enumerate(data):
        x[i, :, :] = scaler.transform(datum)
    x = Variable(torch.from_numpy(x.reshape((x.shape[0], 1, ) + x.shape[1:])).float(),
            volatile=volatile)
    if torch.has_cudnn:
        x = x.cuda()
    return x


def get_output(data, old_dataset=True):
    if old_dataset:
        for i in range(len(data)):
            data[i] = data[i].reshape(data[i].shape[1:])

    y_one_hot = np.zeros((data.shape[0], ) + data[0].shape)
    y_categorical = np.zeros((data.shape[0], ) + data[0].shape[0:1])

    for i, datum in enumerate(data):
        y_one_hot[i, :, :] = datum
        non_zeros = [np.nonzero(dd) for dd in datum]
        non_zeros = [n[0][0] for n in non_zeros]
        y_categorical[i, :] = non_zeros

    y_one_hot = Variable(torch.from_numpy(y_one_hot).float(), requires_grad=False)
    y_categorical = Variable(torch.from_numpy(y_categorical).long(), requires_grad=False)
    if torch.has_cudnn:
        y_one_hot = y_one_hot.cuda()
        y_categorical = y_categorical.cuda()
    return y_one_hot, y_categorical


def main():
    # Getting configuration file from the command line argument
    import importlib
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('checkpoint_path')
    args = parser.parse_args()

    config = importlib.import_module(args.config_file)

    # The common feature extraction layer
    common_feature_extractor = CommonFeatureExtractor(
        out_channels=config.common_out_channels,
        kernel_size=config.common_kernel_size,
        stride=config.common_stride,
        padding=config.common_padding,
        dropout=config.common_dropout,
        dilation=config.common_dilation,
        activation=config.common_activation
    )

    # The alarm branch layers
    branch_alarm = CategoryBranch2(
        cnn_channels_in=config.common_out_channels,
        cnn_channels_out=config.branch_alarm_channels_out,
        cnn_kernel_sizes=config.branch_alarm_cnn_kernel_sizes,
        cnn_strides=config.branch_alarm_cnn_strides,
        cnn_paddings=config.branch_alarm_cnn_paddings,
        cnn_activations=config.branch_alarm_cnn_activations,
        max_pool_kernels=config.branch_alarm_pool_kernels,
        max_pool_strides=config.branch_alarm_pool_strides,
        max_pool_paddings=config.branch_alarm_pool_paddings,
        rnn_input_size=config.branch_alarm_rnn_input_size,
        rnn_out_dims=config.branch_alarm_rnn_output_dims,
        rnn_activations=config.branch_alarm_rnn_activations,
        dropout_cnn=config.branch_alarm_dropout_cnn,
        dropout_rnn_input=config.branch_alarm_dropout_rnn_input,
        dropout_rnn_recurrent=config.branch_alarm_dropout_rnn_recurrent,
        rnn_subsamplings=config.branch_alarm_rnn_subsamplings,
        decoder_dim=config.branch_alarm_decoder_dim,
        output_classes=len(alarm_classes)
    )

    # The vehicle branch layers
    branch_vehicle = CategoryBranch2(
        cnn_channels_in=config.common_out_channels,
        cnn_channels_out=config.branch_vehicle_channels_out,
        cnn_kernel_sizes=config.branch_vehicle_cnn_kernel_sizes,
        cnn_strides=config.branch_vehicle_cnn_strides,
        cnn_paddings=config.branch_vehicle_cnn_paddings,
        cnn_activations=config.branch_vehicle_cnn_activations,
        max_pool_kernels=config.branch_vehicle_pool_kernels,
        max_pool_strides=config.branch_vehicle_pool_strides,
        max_pool_paddings=config.branch_vehicle_pool_paddings,
        rnn_input_size=config.branch_vehicle_rnn_input_size,
        rnn_out_dims=config.branch_vehicle_rnn_output_dims,
        rnn_activations=config.branch_vehicle_rnn_activations,
        dropout_cnn=config.branch_vehicle_dropout_cnn,
        dropout_rnn_input=config.branch_vehicle_dropout_rnn_input,
        dropout_rnn_recurrent=config.branch_vehicle_dropout_rnn_recurrent,
        rnn_subsamplings=config.branch_vehicle_rnn_subsamplings,
        decoder_dim=config.branch_vehicle_decoder_dim,
        output_classes=len(vehicle_classes)
    )

    # Check if we have GPU, and if we do then GPU them all
    if torch.has_cudnn:
        common_feature_extractor = common_feature_extractor.cuda()
        branch_alarm = branch_alarm.cuda()
        branch_vehicle = branch_vehicle.cuda()

    # Create optimizer for all parameters
    params = []
    for block in (common_feature_extractor, branch_alarm, branch_vehicle):
        params += [p for p in block.parameters()]
    optim = torch.optim.Adam(params)

    if os.path.isfile('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        train_data, _ = get_data_stream(
            dataset_name=config.dataset_full_path,
            batch_size=config.batch_size,
            calculate_scaling_metrics=False)
    else:
        train_data, scaler = get_data_stream(
            dataset_name=config.dataset_full_path,
            batch_size=config.batch_size,
            calculate_scaling_metrics=True)
        # Serialize scaler so we don't need to do this again
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # Get the validation data stream
    valid_data, _ = get_data_stream(
        dataset_name=config.dataset_full_path,
        batch_size=config.batch_size,
        that_set='test',
        calculate_scaling_metrics=False,
    )

    for epoch in range(config.epochs):

        for iteration, batch in enumerate(train_data.get_epoch_iterator()):
            # Get input
            x = get_input(batch[0], scaler)

            # Get target values for alarm classes
            y_alarm_1_hot, y_alarm_logits = get_output(batch[-2])

            # Get target values for vehicle classes
            y_vehicle_1_hot, y_vehicle_logits = get_output(batch[-1])

            # Go through the common feature extractor
            common_features = common_feature_extractor(x)

            # Go through the alarm branch
            alarm_output, alarm_weights = branch_alarm(common_features, y_alarm_logits.size(1))

            # Go through the vehicle branch
            vehicle_output, vehicle_weights = branch_vehicle(common_features, y_vehicle_logits.size(1))

            # Calculate losses, do backward passing, and do updates
            loss = total_cost((alarm_output, vehicle_output),
                    (y_alarm_logits, y_vehicle_logits))

            optim.zero_grad()
            loss.backward()
            optim.step()

            print('Epoch {}/it. {}: loss = {}'.format(epoch, iteration, loss.data[0]))

        valid_loss = 0.0
        valid_batches = 0
        for batch in valid_data.get_epoch_iterator():
            # Get input
            x = get_input(batch[0], scaler, volatile=True)

            # Get target values for alarm classes
            y_alarm_1_hot, y_alarm_logits = get_output(batch[-2])

            # Get target values for vehicle classes
            y_vehicle_1_hot, y_vehicle_logits = get_output(batch[-1])

            # Go through the common feature extractor
            common_features = common_feature_extractor(x)

            # Go through the alarm branch
            alarm_output, alarm_weights = branch_alarm(common_features)

            # Go through the vehicle branch
            vehicle_output, vehicle_weights = branch_vehicle(common_features)

            # Calculate validation losses
            valid_loss += total_cost((alarm_output, vehicle_output),
                    (y_alarm_logits, y_vehicle_logits))
            valid_batches += 1

        print('Epoch {}: valid. loss = {}'.format(epoch, valid_loss/valid_batches))


if __name__ == '__main__':
    main()

# EOF
