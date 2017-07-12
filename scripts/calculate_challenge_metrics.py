#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import sed_eval


__author__ = 'Konstantinos Drossos -- TUT'
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


def calculate_tagging_metrics(y_pred_dict, y_true_dict, all_labels, file_prefix=''):
    """

    :param y_pred_dict:
    :type y_pred_dict: dict[str, dict[str, list[str|float]]]
    :param y_true_dict:
    :type y_true_dict: dict[str, dict[str, list[str|float]]]
    :param all_labels:
    :type all_labels: list[str]
    :param file_prefix:
    :type file_prefix: str
    :return: The metrics
    :rtype:
    """

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=all_labels,
                                                                     time_resolution=1)

    for file_id_pred, file_id_true in zip(y_pred_dict.keys(), y_true_dict.keys()):
        file_data_pred = y_pred_dict[file_id_pred]
        file_data_true = y_true_dict[file_id_true]

        pred_data = []
        for event_label, times in file_data_pred.items():
            pred_data.append({
                'event_label': event_label,
                'event_onset': times[0],
                'event_offset': times[1]
            })

        true_data = []
        for event_label, times in file_data_true.items():
            true_data.append({
                'event_label': event_label,
                'event_onset': times[0],
                'event_offset': times[1]
            })

        segment_based_metrics.evaluate(true_data, pred_data)

    # Or print all metrics as reports
    return segment_based_metrics


def main():
    labels = ['car', 'bus', 'truck', 'bike']
    nb_files = 10

    import numpy as np

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

    print(calculate_tagging_metrics(dict_pred, dict_true, labels))


if __name__ == '__main__':
    main()

# EOF
