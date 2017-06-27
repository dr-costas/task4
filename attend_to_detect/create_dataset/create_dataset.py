#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import os
import librosa
import csv
import numpy as np
import h5py
from fuel.datasets import H5PYDataset

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


training_file = 'groundtruth_weak_label_training_set.csv'
testing_file_weak_labels = 'groundtruth_weak_label_testing_set.csv'
testing_file_strong_labels = 'groundtruth_strong_label_testing_set.csv'
fuel_dataset_file = 'dcase_2017_task_4.hdf5'
audio_files_dir = 'audio_files'

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


def extract_features(file_name, n_fft=1024, hop_length=512, n_mels=64):
    """

    :param file_name:
    :type file_name: str
    :param n_fft:
    :type n_fft: int
    :param hop_length:
    :type hop_length: int
    :param n_mels:
    :type n_mels: int
    :return: np.ndarray [shape=(t, n_mels + delta_mels + delta_delta_mels)]
    :rtype: numpy.core.multiarray.ndarray
    """
    y, sr = librosa.core.load(file_name, sr=44100)
    magn_spectr = np.abs(librosa.stft(y + np.spacing([1])[0], n_fft=n_fft, hop_length=hop_length))
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_band = np.dot(mel_filters, magn_spectr)
    log_mel_band = np.log(mel_band + np.spacing([1])[0])
    delta_mels = librosa.feature.delta(log_mel_band, width=7, order=1)
    delta_delta_mels = librosa.feature.delta(log_mel_band, width=7, order=2)

    return np.concatenate((log_mel_band, delta_mels, delta_delta_mels)).T


def make_dict_from_csv(csv_file_name):
    """

    :param csv_file_name:
    :type csv_file_name: str
    :return:
    :rtype: dict[str, dict[str, float|list[str]]]
    """
    to_return = {}
    with open(csv_file_name) as f:
        reader = csv.reader(f)

        for row in reader:
            to_return.update({
                row[0].strip(): {
                    'start': float(row[1]),
                    'end': float(row[2]),
                    'classes_alarm': [s.strip() for s in row[3].split(',') if s in alarm_classes],
                    'classes_vehicle': [s.strip() for s in row[3].split(',') if s in vehicle_classes]
                }
            })

    return to_return


def get_yt_name(file_name):
    """

    :param file_name:
    :type file_name: str
    :return:
    :rtype: str
    """
    return ''.join(os.path.split(file_name)[-1].split('_')[:-2])[1:]


def process_data(audio_files, set_dict):
    """

    :param audio_files:
    :type audio_files: list[str]
    :param set_dict:
    :type set_dict: dict[str, dict[str, float|list[str]]]
    :return:
    :rtype: list[dict[str, numpy.core.multiarray.ndarray]]
    """
    to_return = []
    nb_classes_alarm = len(alarm_classes)
    nb_classes_vehicle = len(vehicle_classes)

    for audio_file in audio_files:
        features = extract_features(audio_file).astype('float32')
        classes_alarm = np.zeros((nb_classes_alarm + 1, nb_classes_alarm + 1)).astype('uint8')
        classes_vehicle = np.zeros((nb_classes_vehicle + 1, nb_classes_vehicle + 1)).astype('uint8')
        yt_file = get_yt_name(audio_file)
        cl_a = set_dict[yt_file]['classes_alarm']
        cl_v = set_dict[yt_file]['classes_vehicle']
        indices_a = [alarm_classes.index(c) for c in cl_a]
        indices_v = [alarm_classes.index(c) for c in cl_v]
        for i in indices_a:
            classes_alarm[i, i] = 1
        for i in indices_v:
            classes_vehicle[i, i] = 1

        to_return.append({
            'audio_features': features,
            'classes_alarm': classes_alarm,
            'classes_vehicle': classes_vehicle,
        })

    return to_return


def prepare_data(for_set):
    """

    :param for_set:
    :type for_set: str
    :return:
    :rtype: (list[dict[str, numpy.core.multiarray.ndarray]], \
            list[dict[str, numpy.core.multiarray.ndarray]],
            list[dict[str, numpy.core.multiarray.ndarray]])
    """
    # Get all wav files
    all_files_training = [os.path.abspath(f) for f in os.listdir(audio_files_dir.format(training_file))]
    set_dict = make_dict_from_csv(for_set)
    return process_data(all_files_training, set_dict)


def make_fuel_dataset(file_name, training_set, testing_set_weak, testing_set_strong):
    """

    :param file_name:
    :type file_name:
    :param training_set:
    :type training_set: list[dict[str, numpy.core.multiarray.ndarray]]
    :param testing_set_weak:
    :type testing_set_weak: list[dict[str, numpy.core.multiarray.ndarray]]
    :param testing_set_strong:
    :type testing_set_strong: list[dict[str, numpy.core.multiarray.ndarray]]
    :return:
    :rtype:
    """
    f = h5py.File(file_name, mode='w')

    nb_training_examples = len(training_set)
    nb_testing_examples = len(testing_set_weak)
    nb_examples = nb_training_examples + nb_testing_examples

    # Audio features
    audio_features = f.create_dataset(
        'audio_features', (nb_examples, ),
        dtype=h5py.special_dtype(vlen=np.dtype('float32'))
    )

    audio_features[...] = [entry['audio_features'].flatten()
                           for entry in training_set + testing_set_weak]

    audio_features_shapes = f.create_dataset(
        'audio_features_shapes', (nb_examples, 2), dtype='int32')
    audio_features_shapes[...] = np.array([
        entry['audio_features'].reshape((1, ) + entry['audio_features'].shape).shape
        for entry in training_set + testing_set_weak]
    )
    audio_features.dims.create_scale(audio_features_shapes, 'shapes')
    audio_features.dims[0].attach_scale(audio_features_shapes)

    audio_features_shape_labels = f.create_dataset('audio_features_shape_labels', (3, ), dtype='S11')
    audio_features_shape_labels[...] = [
        'batch'.encode('utf8'), 'time_frames'.encode('utf8'), 'features'.encode('utf8')
    ]
    audio_features.dims.create_scale(audio_features_shape_labels, 'shape_labels')
    audio_features.dims[0].attach_scale(audio_features_shape_labels)

    # Weak labels
    weak_labels_datasets = []
    tuples_weak = [('targets_vehicle_weak', 'classes_vehicle'),
                   ('targets_alarm_weak', 'classes_alarm')]

    for t in tuples_weak:
        d = f.create_dataset(t[0], (nb_examples, ) + training_set[0][t[1]].shape, dtype='unit8')
        shape_vehicle_weak = (1,) + training_set[0][t[1]].shape
        d[...] = np.vstack([pair[t[1]].reshape(shape_vehicle_weak)
                            for pair in training_set + testing_set_weak])

        for i, label in enumerate(['batch', 'classes_frames', 'targets']):
            d.dims[i] = label

        weak_labels_datasets.append(d)

    # Strong labels
    strong_labels_datasets = []
    tuples_strong = [('targets_vehicle_strong', 'classes_vehicle'),
                     ('targets_alarm_strong', 'classes_alarm')]

    for t in tuples_strong:
        d = f.create_dataset(t[0], (nb_examples, ) + testing_set_strong[0][t[1]].shape, dtype='unit8')
        shapes = (1,) + testing_set_strong[0][t[1]].shape
        d[...] = np.vstack([np.zeros(shapes) for _ in training_set] +
                           [pair[t[1]].reshape(shapes) for pair in testing_set_strong])

        for i, label in enumerate(['batch', 'time_frames', 'targets']):
            d[i].label = label

        strong_labels_datasets.append(d)

    split_dict = {
        'train': {
            'audio_features': (0, nb_training_examples),
            'targets_weak': (0, nb_training_examples),
            'targets_strong': (0, nb_training_examples)
        },
        'test': {
            'audio_features': (nb_training_examples, nb_examples),
            'targets_weak': (nb_training_examples, nb_examples),
            'targets_strong': (nb_training_examples, nb_examples)
        }
    }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()


def main():
    training, testing_weak, testing_strong = prepare_data(training_file)
    make_fuel_dataset(fuel_dataset_file, training, testing_weak, testing_strong)


if __name__ == '__main__':
    main()

# EOF
