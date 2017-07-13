#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from attend_to_detect.evaluation import (
    tagging_metrics_from_raw_output, tagging_metrics_from_dictionaries)


__docformat__ = 'reStructuredText'


def make_sed_txt_file(values_dict, file_name):
    """

    :param values_dict: A dictionary with each key be a different sound event class \
                        and each value of the key be the a list of float (or str) \
                        that has the start and ending times for the sound event
    :type values_dict: dict[str, list[float|str]]
    :param file_name: The full file path (name and extension included) for the \
                      resulting file.
    :type file_name: str
    """
    with open(file_name, 'w') as f:
        for k, v in values_dict.items():
            f.write('{s_time}\t{e_time}\t{class_name}\n'.format(
                class_name=k,
                s_time=v[0],
                e_time=v[1]
            ))


def make_tagging_txt_file(values_dict, file_name):
    """

    :param values_dict: A dictionary with each key be a different example ID and\
                        each value of the key be the tag that corresponds to that\
                        file.
    :type values_dict: dict[str, str]
    :param file_name: The full file path (name and extension included) for the \
                      resulting file.
    :type file_name: str
    """
    for k, v in values_dict.items():
        with open(file_name, 'w') as f:
            f.write('0.000\t10.000\t{}\n'.format(v))
        break


def calculate_sed_metrics(y_ped_dict, y_true_dict):
    pass


def main():
    labels = ['car', 'bus', 'truck', 'bike']
    nb_files = 10

    pred_indices = np.random.randint(0, len(labels), nb_files)
    true_indices = np.random.randint(0, len(labels), nb_files)

    dict_pred = {}
    dict_true = {}

    for i, (i_pred, i_true) in enumerate(zip(pred_indices, true_indices)):
        dict_pred.update({
            i: {labels[i_pred]: [0.00, 10.00]}
        })
        dict_true.update({
            i: {labels[i_true]: [0.00, 10.00]}
        })

    print(tagging_metrics_from_dictionaries(dict_pred, dict_true, labels))

    x = np.random.random((nb_files, 3, len(labels)))
    y_categorical = np.random.randint(0, len(labels), (nb_files, 3))
    y_1_hot = np.zeros((nb_files, 3, len(labels)))
    for i in range(nb_files):
        for ii in range(3):
            random_col = np.random.randint(0, len(labels))
            y_1_hot[i, ii, random_col] = 1

    print(tagging_metrics_from_raw_output(x, y_categorical, labels, y_true_is_categorical=True))
    print(tagging_metrics_from_raw_output(x, y_1_hot, labels, y_true_is_categorical=False))


if __name__ == '__main__':
    main()
