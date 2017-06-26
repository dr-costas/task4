from __future__ import absolute_import
from __future__ import print_function

import numpy as np
np.random.seed(42)  # for reproducibility

from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from stft_dataset import STFTDataset
from model import RNNModelWithSkipConnections, train_fn, valid_fn
from pytorch_utils import TrainLoop, load_checkpoint

def load_data(window_size, step_size, train_file, valid_file):
    print("Loading data...")
    G_train = STFTDataset(window=window_size, step=step_size)
    G_train.load_metadata_from_desc_file(train_file)
    G_train.fit_stats()

    G_val = STFTDataset(window=window_size, step=step_size)
    G_val.load_metadata_from_desc_file(valid_file)
    G_val.feats_mean = G_train.feats_mean
    G_val.feats_std = G_train.feats_std

    return G_train, G_val


if __name__ == '__main__':

    from argparse import ArgumentParser
    import os
    from glob import glob

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--n_readouts', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--step_size', type=int, default=16)
    parser.add_argument('--time_context', type=int, default=11)
    parser.add_argument('--freq_context', type=int, default=21)
    parser.add_argument('--train_file', default='train.json')
    parser.add_argument('--valid_file', default='valid.json')
    parser.add_argument('checkpoint_path')

    args = parser.parse_args()

    try:
        train_loop = load_checkpoint(args.checkpoint_path)
    except ValueError:
        print('No checkpoints, initializing a model from scratch...')
        window_size = args.window_size # in ms
        step_size = args.step_size
        n_input = int(1e-3*window_size*16000/2 + 1)
        n_output = n_input

        model = RNNModelWithSkipConnections(n_input,
                args.n_hidden,
                args.n_readouts,
                n_output,
                context=(args.freq_context, args.time_context)).cuda()

        #FIXME: criterion has to make sense for our task
        criterion = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        G_train, G_val = load_data(window_size, step_size, train_file=args.train_file, valid_file=args.valid_file)

        train_loader = DataLoader(G_train, batch_size=args.batch_size,
                num_workers=4, shuffle=True)
        valid_loader = DataLoader(G_val, batch_size=args.batch_size,
                num_workers=2)

        train_loop = TrainLoop(model,
                    optimizer, criterion,
                    train_fn, train_loader,
                    valid_fn, valid_loader,
                    checkpoint_path=args.checkpoint_path)

    train_loop.train(args.max_epochs)

