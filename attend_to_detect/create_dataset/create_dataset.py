#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import absolute_import
from .download_audio import main as download_audio_main
import os
import shutil
import librosa
import csv
import numpy as np

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


training_file = 'training_set.csv'
testing_file = 'testing_set.csv'

training_file_ground_truth = 'groundtruth_weak_label_training_set.csv'
testing_file_ground_truth_weak = 'groundtruth_weak_label_testing_set.csv'
testing_file_ground_truth_strong = 'groundtruth_strong_label_testing_set.csv'

audio_dirs_to_delete = [
    '{file_name}_{file_name}_audio_downloaded',
    '{file_name}_{file_name}_audio_formatted_downloaded'
]

audio_files_dir = '{file_name}_{file_name}_audio_formatted_and_segmented_downloads'

alarm_classes = [
    'Train horn',
    'Truck horn',
    'Car alarm',
    'Reversing beeps',
    'Ambulance siren',
    'Police siren',
    'Fire truck siren',
    'Civil defense siren',
    'Screaming'
]

vehicle_classes = [
    'Bicycle',
    'Skateboard',
    'Car',
    'Car passing',
    'Bus',
    'Truck',
    'Motorcycle',
    'Train'
]


def download_dataset():
    for data_split in [training_file, testing_file]:
        download_audio_main(data_split)
        [shutil.rmtree(dir_to_delete.format(file_name=os.path.splitext(data_split)[0]))
         for dir_to_delete in audio_dirs_to_delete]


def extract_features(file_name, n_fft=1024, hop_length=512, power=1.0, n_mels=64):
    """

    :param file_name:
    :type file_name: str
    :param n_fft:
    :type n_fft: int
    :param hop_length:
    :type hop_length: int
    :param power:
    :type power: float
    :param n_mels:
    :type n_mels: int
    :return: np.ndarray [shape=(t, n_mels + delta_mels + delta_delta_mels)]
    :rtype: numpy.core.multiarray.ndarray
    """
    y, sr = librosa.core.load(file_name, sr=44100)
    mel_band_energies = librosa.feature.melspectrogram(
        y=y, sr=sr, S=None,
        n_fft=n_fft, hop_length=hop_length, power=power,
        n_mels=n_mels,
    )
    delta_mels = librosa.feature.delta(mel_band_energies, width=7, order=1)
    delta_delta_mels = librosa.feature.delta(mel_band_energies, width=7, order=2)

    return np.concatenate((mel_band_energies, delta_mels, delta_delta_mels)).T


def get_training_set():

    features = []
    classes = []

    with open(training_file_ground_truth) as f:
        reader = csv.reader(f)

        for row in reader:
            file_name = row[0].split('\t')[0]
            class_name = row[0].split('\t')[-1]

            file_name = os.path.join(
                audio_files_dir.format(training_file),
                file_name
            )

            features.append(extract_features(file_name))
            classes.append(class_name)


def create_fuel_dataset():
    pass


def main():
    download_dataset()


if __name__ == '__main__':
    main()

# EOF
