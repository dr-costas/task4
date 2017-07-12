#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import importlib
import numpy as np
import timeit
from contextlib import closing
from argparse import ArgumentParser
from mimir import Logger
from tqdm import tqdm

import torch
from torch.nn import functional
from attend_to_detect.dataset import vehicle_classes, alarm_classes, get_input, get_output, get_data_stream
from attend_to_detect.model import CategoryBranch2, CommonFeatureExtractor

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


def category_cost(out_hidden, target):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
    target_flat = target.view(-1)
    return torch.nn.functional.cross_entropy(out_hidden_flat, target_flat)


def total_cost(hiddens, targets):
    return category_cost(hiddens[0], targets[0]) + \
            category_cost(hiddens[1], targets[1])


def main():
    # Getting configuration file from the command line argument
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
        output_classes=len(alarm_classes) + 1
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
        output_classes=len(vehicle_classes) + 1
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

    logger = Logger("{}_log.jsonl.gz".format(args.checkpoint_path),
                    formatter=None)
    with closing(logger):
        train_loop(
            config, common_feature_extractor, branch_vehicle, branch_alarm,
            train_data, valid_data, scaler, optim, logger)


def train_loop(config, common_feature_extractor, branch_vehicle, branch_alarm,
               train_data, valid_data, scaler, optim, logger):
    total_iterations = 0
    for epoch in range(config.epochs):
        common_feature_extractor.train()
        branch_alarm.train()
        branch_vehicle.train()
        losses_alarm = []
        losses_vehicle = []
        epoch_start_time = timeit.timeit()
        for iteration, batch in tqdm(enumerate(train_data.get_epoch_iterator()), total=50000):
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
            loss_a = category_cost(alarm_output, y_alarm_logits)
            loss_v = category_cost(vehicle_output, y_vehicle_logits)
            loss = loss_a + loss_v

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses_alarm.append(loss_a.data[0])
            losses_vehicle.append(loss_v.data[0])

            if total_iterations % 10 == 0:
                logger.log({'iteration': total_iterations,
                            'epoch': epoch,
                            'train': {'alarm_loss': loss_a.data[0],
                                      'vehicle_loss': loss_v.data[0]}})

            total_iterations += 1

        print('Epoch {:4d} elapsed training time {:10.5f}'
              '\tLosses: alarm: {:10.6f} | vehicle: {:10.6f}'.format(
                epoch, epoch_start_time - timeit.timeit(),
                np.mean(losses_alarm), np.mean(losses_vehicle)))

        common_feature_extractor.eval()
        branch_alarm.eval()
        branch_vehicle.eval()
        valid_batches = 0
        loss_a = 0.0
        loss_v = 0.0
        validation_start_time = timeit.timeit()
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
            alarm_output, alarm_weights = branch_alarm(common_features, y_alarm_logits.size(1))

            # Go through the vehicle branch
            vehicle_output, vehicle_weights = branch_vehicle(common_features, y_vehicle_logits.size(1))

            # Calculate validation losses
            loss_a += category_cost(alarm_output, y_alarm_logits).data[0]
            loss_v += category_cost(vehicle_output, y_vehicle_logits).data[0]

            valid_batches += 1

        print('Epoch {:4d} validation elapsed time {:10.5f}'
              '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, validation_start_time - timeit.timeit(),
                loss_a/valid_batches, loss_v/valid_batches))
        logger.log({'iteration': total_iterations,
                    'epoch': epoch,
                    'valid': {'alarm_loss': loss_a/valid_batches,
                              'vehicle_loss': loss_v/valid_batches}})


if __name__ == '__main__':
    main()
