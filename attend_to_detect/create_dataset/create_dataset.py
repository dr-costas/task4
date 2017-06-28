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


training_file = 'training_set_test.csv'
testing_file_weak_labels = 'testing_set_test.csv'
testing_file_strong_labels = 'groundtruth_strong_label_testing_set_test.csv'
fuel_dataset_file = 'dcase_2017_task_4.hdf5'
audio_files_dir_training = 'audio_files_small_tests'
audio_files_dir_testing = 'audio_files_testing'

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


def make_dicts_from_csvs(csv_file_names):
    """

    :param csv_file_names: [training_csv, testing_weak_csv, testing_strong_csv]
    :type csv_file_names: list[str]
    :return: [training_set, testing_set]
    :rtype: list[dict[str, dict[str, float|list[str]]]]
    """
    to_return_training = {}
    with open(csv_file_names[0]) as f:
        reader = csv.reader(f)

        for row in reader:
            classes_alarm = []
            classes_vehicle = []
            for s in row[3].split(','):
                s_strip = s.strip()
                if s_strip in alarm_classes:
                    classes_alarm.append(s_strip)
                elif s_strip in vehicle_classes:
                    classes_vehicle.append(s_strip)
            to_return_training.update({
                row[0].strip(): {
                    'start': float(row[1]),
                    'end': float(row[2]),
                    'classes_alarm_weak': classes_alarm,
                    'classes_vehicle_weak': classes_vehicle,
                    'classes_alarm_strong_labels': [],
                    'classes_alarm_strong_times': [],
                    'classes_vehicle_strong_labels': [],
                    'classes_vehicle_strong_times': []
                }
            })

    to_return_testing = {}
    with open(csv_file_names[1]) as f, open(csv_file_names[2]) as f_2:
        reader = csv.reader(f)
        reader_2 = csv.reader(f_2, delimiter='\t')

        for row in reader:
            classes_alarm = []
            classes_vehicle = []
            for s in row[3].split(','):
                s_strip = s.strip()
                if s_strip in alarm_classes:
                    classes_alarm.append(s_strip)
                elif s_strip in vehicle_classes:
                    classes_vehicle.append(s_strip)
            to_return_testing.update({
                row[0].strip(): {
                    'start': float(row[1]),
                    'end': float(row[2]),
                    'classes_alarm_weak': classes_alarm,
                    'classes_vehicle_weak': classes_vehicle,
                    'classes_alarm_strong_labels': [],
                    'classes_alarm_strong_times': [],
                    'classes_vehicle_strong_labels': [],
                    'classes_vehicle_strong_times': []
                }
            })

        for row in reader_2:
            start_time = round(float(row[1]))
            end_time = round(float(row[2]))
            yt_id = ''.join(row[0].split('_')[:-2]).strip()
            for s in row[3].split(','):
                s_strip = s.strip()
                if s_strip in alarm_classes:
                    to_return_testing[yt_id]['classes_alarm_strong_labels'].append(s_strip)
                    to_return_testing[yt_id]['classes_alarm_strong_times'].append([start_time, end_time])
                elif s_strip in vehicle_classes:
                    to_return_testing[yt_id]['classes_vehicle_strong_labels'].append(s_strip)
                    to_return_testing[yt_id]['classes_vehicle_strong_times'].append([start_time, end_time])

    return [to_return_training, to_return_testing]


def get_yt_name(file_name):
    """

    :param file_name:
    :type file_name: str
    :return:
    :rtype: str
    """
    return ''.join(os.path.split(file_name)[-1].split('_')[:-2])[1:]


def process_data(audio_files, sets_dict):
    """

    :param audio_files: [training_set, testing_set]
    :type audio_files: list[list[str]]
    :param sets_dict: [training_set, testing_set]
    :type sets_dict: list[dict[str, dict[str, float|list[str]]]]
    :return:
    :rtype: (list[dict[str, numpy.core.multiarray.ndarray]], \
            list[dict[str, numpy.core.multiarray.ndarray]])
    """
    def process_set(set_files, inner_set_dict):
        to_return_inner = []
        for set_file in set_files:
            features = extract_features(set_file).astype('float32')
            yt_file = get_yt_name(set_file)
            cl_a_w = inner_set_dict[yt_file]['classes_alarm_weak']
            cl_a_s = inner_set_dict[yt_file]['classes_alarm_strong']
            times_a_s = inner_set_dict[yt_file]['classes_alarm_strong_times']
            cl_v_w = inner_set_dict[yt_file]['classes_vehicle_weak']
            cl_v_s = inner_set_dict[yt_file]['classes_vehicle_strong']
            times_v_s = inner_set_dict[yt_file]['classes_vehicle_strong_times']

            indices_a_w = [alarm_classes.index(c) + 1 for c in cl_a_w] + [0]
            indices_v_w = [vehicle_classes.index(c) + 1 for c in cl_v_w] + [0]

            strong_a = [(alarm_classes.index(c) + 1, ) + t for c, t in zip(cl_a_s, times_a_s)]
            strong_v = [(vehicle_classes.index(c) + 1, ) + t for c, t in zip(cl_v_s, times_v_s)]

            to_return_inner.append({
                'audio_features': features,
                'classes_alarm_weak': indices_a_w,
                'classes_vehicle_weak': indices_v_w,
                'classes_alarm_strong': strong_a,
                'classes_vehicle_strong': strong_v,
            })

        return to_return_inner

    return process_set(audio_files[0], sets_dict[0]), process_set(audio_files[1], sets_dict[1])


def prepare_data(set_files):
    """

    :param set_files:
    :type set_files: list[str]
    :return:
    :rtype: (list[dict[str, numpy.core.multiarray.ndarray]], \
            list[dict[str, numpy.core.multiarray.ndarray]],
            list[dict[str, numpy.core.multiarray.ndarray]])
    """
    # Get all wav files
    sets_dict = make_dicts_from_csvs(set_files)
    all_training = [os.path.abspath(f) for f in os.listdir(audio_files_dir_training)]
    all_testing = [os.path.abspath(f) for f in os.listdir(audio_files_dir_testing)]

    return process_data([all_training, all_testing], sets_dict)


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
            'targets_vehicle_weak': (0, nb_training_examples),
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
    set_files = [
        training_file,
        testing_file_weak_labels,
        testing_file_strong_labels
    ]
    training, testing_weak, testing_strong = prepare_data(set_files)
    make_fuel_dataset(fuel_dataset_file, training, testing_weak, testing_strong)


if __name__ == '__main__':
    main()

# EOF
