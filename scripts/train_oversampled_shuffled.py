#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import importlib
import numpy as np
import timeit
import shutil
from argparse import ArgumentParser
from contextlib import closing
from mimir import Logger
from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from attend_to_detect.dataset import (
    vehicle_classes, alarm_classes, get_input, get_output_binary_one_hot, get_data_stream_single,
    get_output_binary_single)
from attend_to_detect.pytorch_dataset import ChallengeDataset
from attend_to_detect.model import CategoryBranch2
from attend_to_detect.evaluation import category_cost, tagging_metrics_from_list

__docformat__ = 'reStructuredText'

def validate(valid_data, network, logger, total_iterations, epoch):
    valid_batches = 0
    loss = 0.0
    accuracy = 0.0

    validation_start_time = timeit.timeit()
    predictions = []
    ground_truths = []
    for batch in valid_data.get_epoch_iterator():
        x, y = batch
        x = torch.autograd.Variable(x.cuda())
        y = torch.autograd.Variable(y.cuda(), requires_grad=False)

        output, attention_weights = network(x, y.size(1))

        loss += category_cost(output, y).data[0]
        valid_batches += 1

        if torch.has_cudnn:
            output = output.cpu()

        predictions.extend(predictions_to_list(output))
        ground_truths.extend(predictions_to_list(y, targets=True))
    print('Epoch {:4d} validation elapsed time {:10.5f} sec(s) | Valid. loss alarm: {:10.6f}'.format(
                epoch, timeit.timeit() - validation_start_time,
                loss/valid_batches))
    metrics = tagging_metrics_from_list(
        predictions, ground_truths, alarm_classes + vehicle_classes)
    print(metrics)

    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'validation': dict(
                        loss=loss/valid_batches,
                    )
                }})


def predictions_to_list(pred, targets=False):
    batch_size = pred.size(0)
    n_timesteps = pred.size(1)
    if targets:
        classes = pred - 1
    else:
        pred_flat = torch.nn.functional.softmax(pred.view(-1, pred.size(2)))
        classes_flat = torch.max(pred_flat, -1)[1]
        classes = classes_flat.view(batch_size, n_timesteps, -1) - 1
    pred_list = []
    all_classes = alarm_classes + vehicle_classes
    for sample in classes:
        y = []
        for n in range(n_timesteps):
            if sample[n].data[0] >= 0:
                event = dict(event_offset=10.00,
                        event_onset=00.00,
                        event_label=all_classes[sample[n].data[0]])
                y.append(event)
        pred_list.append(y)
    return pred_list

def main():
    # Getting configuration file from the command line argument
    parser = ArgumentParser()
    parser.add_argument('--train-examples', type=int, default=-1)
    parser.add_argument('config_file')
    parser.add_argument('checkpoint_path')
    parser.add_argument('--print-grads', action='store_true')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--visdom-port', type=int, default=5004)
    parser.add_argument('--visdom-server', default='http://localhost')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--job-id', default='')
    args = parser.parse_args()

    if args.debug:
        torch.has_cudnn = False

    config = importlib.import_module(args.config_file)

    # The main model
    network = CategoryBranch2(
        cnn_channels_in=1,
        cnn_channels_out=config.network_channels_out,
        cnn_kernel_sizes=config.network_cnn_kernel_sizes,
        cnn_strides=config.network_cnn_strides,
        cnn_paddings=config.network_cnn_paddings,
        cnn_activations=config.network_cnn_activations,
        max_pool_kernels=config.network_pool_kernels,
        max_pool_strides=config.network_pool_strides,
        max_pool_paddings=config.network_pool_paddings,
        rnn_input_size=config.network_rnn_input_size,
        rnn_out_dims=config.network_rnn_output_dims,
        rnn_activations=config.network_rnn_activations,
        dropout_cnn=config.network_dropout_cnn,
        dropout_rnn_input=config.network_dropout_rnn_input,
        dropout_rnn_recurrent=config.network_dropout_rnn_recurrent,
        rnn_subsamplings=config.network_rnn_subsamplings,
        decoder_dim=config.network_decoder_dim,
        output_classes=len(alarm_classes) + len(vehicle_classes) + 1,
        attention_bias=config.network_attention_bias,
        init=config.network_init
    )

    print('Total parameters: {}'.format(network.nb_trainable_parameters()))

    # Check if we have GPU, and if we do then GPU them all
    if torch.has_cudnn:
        network = network.cuda()

    # Create optimizer for all parameters
    optim = config.optimizer(network.parameters(), lr=config.optimizer_lr)

    # Do we have a checkpoint?
    if os.path.isdir(args.checkpoint_path):
        print('Checkpoint directory exists!')
        if os.path.isfile(os.path.join(args.checkpoint_path, 'latest.pt')):
            print('Loading checkpoint...')
            ckpt = torch.load(os.path.join(args.checkpoint_path, 'latest.pt'))
            network.load_state_dict(ckpt['network'])
            optim.load_state_dict(ckpt['optim'])
    else:
        print('Checkpoint directory does not exist, creating...')
        os.makedirs(args.checkpoint_path)

    if args.train_examples == -1:
        examples = None
    else:
        examples = args.train_examples

    # Load datasets
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('weights.pkl', 'rb') as f:
        weights = pickle.load(f)

    train_dataset = ChallengeDataset(dataset_name='/Tmp/santosjf/dcase_2017_task_4_test.hdf5',
            that_set='train', scaler=scaler, shuffle_targets=True)
    valid_dataset = ChallengeDataset(dataset_name='/Tmp/santosjf/dcase_2017_task_4_test.hdf5',
            that_set='test', scaler=scaler, shuffle_targets=False)

    train_sampler = WeightedRandomSampler(weights, len(train_dataset))
    train_data = DataLoader(train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            collate_fn=ChallengeDataset.collate)
            #num_workers=2)
    train_data.get_epoch_iterator = train_data.__iter__
    valid_data = DataLoader(valid_dataset,
            batch_size=config.batch_size,
            collate_fn=ChallengeDataset.collate)
    valid_data.get_epoch_iterator = valid_data.__iter__

    logger = Logger("{}_log.jsonl.gz".format(args.checkpoint_path),
                    formatter=None)
    if args.visdom:
        from attend_to_detect.utils.visdom_handler import VisdomHandler
        title_losses = 'Train/valid losses'
        title_accu = 'Train/valid accuracies'
        if args.job_id != '':
            title_losses += ' - Job ID: {}'.format(args.job_id)
            title_accu += ' - Job ID: {}'.format(args.job_id)
        loss_handler = VisdomHandler(
            ['training', 'validation'],
            'loss',
            dict(title=title_losses,
                 xlabel='iteration',
                 ylabel='cross-entropy'),
            server=args.visdom_server, port=args.visdom_port)
        logger.handlers.append(loss_handler)
        accuracy_handler = VisdomHandler(
            ['training', 'validation'],
            'acc',
            dict(title=title_accu,
                 xlabel='iteration',
                 ylabel='accuracy, %'),
            server=args.visdom_server, port=args.visdom_port)
        logger.handlers.append(accuracy_handler)

    with closing(logger):
        train_loop(
            config, network,
            train_data, valid_data, scaler, optim, args.print_grads, logger,
            args.checkpoint_path, args.no_tqdm)


def iterate_params(pytorch_module):
    has_children = False
    for child in pytorch_module.children():
        for pair in iterate_params(child):
            yield pair
        has_children = True
    if not has_children:
        for name, parameter in pytorch_module.named_parameters():
            yield (parameter, name, pytorch_module)


def train_loop(config, network, train_data, valid_data, scaler,
               optim, print_grads, logger, checkpoint_path, no_tqdm):
    total_iterations = 0
    # loss_module = torch.nn.MSELoss()
    for epoch in range(config.epochs):
        network.train()
        losses = []
        accuracies = []
        epoch_start_time = timeit.timeit()
        epoch_iterator = enumerate(train_data.get_epoch_iterator())
        if not no_tqdm:
            epoch_iterator = tqdm(epoch_iterator,
                                  total=50000 // config.batch_size)
        for iteration, batch in epoch_iterator:
            x, y = batch
            x = torch.autograd.Variable(x.cuda())
            y = torch.autograd.Variable(y.cuda(), requires_grad=False)
            network_output, attention_weights = network(x, y.size(1))

            # Calculate losses, do backward passing, and do updates
            loss = category_cost(network_output, y)
            #pred_list = predictions_to_list(y, targets=True)

            optim.zero_grad()
            loss.backward()

            if config.grad_clip_norm > 0:
                clip_grad_norm(network, config.grad_clip_norm)

            optim.step()

            if print_grads:
                for param, name, module in iterate_params(network):
                    print("{}\t\t {}\t\t: grad norm {}\t\t weight norm {}".format(
                        name, str(module), param.grad.norm(2).data[0],
                        param.norm(2).data[0]))

            losses.append(loss.data[0])

            #accuracies.append(binary_accuracy_single(network_output, target_values))

            if total_iterations % 10 == 0:
                logger.log({
                    'iteration': total_iterations,
                    'epoch': epoch,
                    'records': {
                        'training': dict(
                            loss=np.mean(losses[-10:]),
                            #acc=np.mean(accuracies[-10:])),
                        )
                    }
                })

            total_iterations += 1

        print('Epoch {:3d} | Elapsed training time {:10.3f} | Loss: {:10.6f}'.format(
                epoch, epoch_start_time - timeit.timeit(),
                np.mean(losses)))

        # Validation
        network.eval()
        validate(valid_data, network, logger, total_iterations, epoch)

        # Checkpoint
        ckpt = {'network': network.state_dict(),
                'optim': optim.state_dict()}
        torch.save(ckpt, os.path.join(checkpoint_path, 'ckpt_{}.pt'.format(epoch)))
        shutil.copyfile(
            os.path.join(checkpoint_path, 'ckpt_{}.pt'.format(epoch)),
            os.path.join(checkpoint_path, 'latest.pt'))


if __name__ == '__main__':
    main()
