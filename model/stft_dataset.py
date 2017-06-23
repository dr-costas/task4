"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function
from functools import reduce

import torch
from torch.utils.data import Dataset
import random
import json
import warnings
import numpy as np
from utils import calc_feat_dim, spectrogram_from_file

class STFTDataset(Dataset):
    def __init__(self, step=10, window=20, max_freq=8000, desc_file=None,
            pad=0):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.pad = pad

    def featurize(self, audio_clip, target=False):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq, pad=self.pad)[0]

    def load_metadata_from_desc_file(self, desc_file):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        audio_paths, targets = [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    targets.append(spec['labels'])
                except Exception as e:
                    warnings.warn('Error reading line #{}: {}'
                                .format(line_num, json_line))
                    warnings.warn(str(e))

        self.audio_paths = audio_paths
        self.targets = targets

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def __getitem__(self, index):
        features = self.normalize(self.featurize(self.audio_paths[index]))
        # TODO: target is a list or tuple with the indices of the classes that
        # should be predicted. We probably have to format it for PyTorch instead
        # of just converting it to a FloatTensor like is done here
        target = self.targets[index]

        features = torch.FloatTensor(features)
        target = torch.FloatTensor(target)

        return features, target

    def __len__(self):
        return len(self.audio_paths)

    def fit_stats(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.audio_paths))
        rng = random.Random(42)
        samples = rng.sample(range(len(self.audio_paths)), k_samples)
        feats = [self.featurize(self.audio_paths[s]) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoaderIter
    from torch.utils.data import DataLoader
    d = STFTDataset(pad=4, normalize_targets=True)
    d.load_metadata_from_desc_file('valid.json')
    d.fit_stats()
    loader = DataLoader(d, 4)
    itr = DataLoaderIter(loader)
    x, y, z = next(itr)

