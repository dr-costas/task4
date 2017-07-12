#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import sed_eval
import numpy as np


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


def tagging_metrics_from_dictionaries(y_pred_dict, y_true_dict, all_labels, file_prefix=''):
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


def tagging_metrics_from_raw_output(y_pred, y_true, all_labels, y_true_is_categorical=True):
    """

    If 1-hot-encoding, the value [1, 0, 0, ..., 0] is considered <EOS>. \
    If not 1-hot-encoded, the value 0 is considered <EOS>.

    :param y_pred: List with predictions, of len = nb_files and each element \
                   is a matrix of size [nb_events_detected x nb_labels]
    :type y_pred: list[numpy.core.multiarray.ndarray]
    :param y_true: List with true values, of len = nb_files. If 1-hot-enconding \
                   each element is a matrix of size [nb_events x nb_labels] . \
                   Else, each element is an array of len = nb_events.
    :type y_true: list[numpy.core.multiarray.ndarray]
    :param all_labels: A list with all the labels **including** the <EOS> at the 0th index
    :type all_labels: list[str]
    :param y_true_is_categorical: Indication if y_true is categorical
    :type y_true_is_categorical: bool
    :return:
    :rtype:
    """
    def get_data(the_data, the_labels, is_categorical):

        dict_to_return = {}
        for i, file_data in enumerate(the_data):
            found_eos = False
            event_labels = []
            for event_data in file_data:
                if is_categorical:
                    index = event_data
                else:
                    index = np.argmax(event_data)

                if index == 0 and not found_eos:
                    found_eos = True
                elif index == 0 and found_eos:
                    break

                event_labels.append(the_labels[np.argmax(event_data)])

            dict_f = {e: [0.00, 10.00] for e in event_labels}
            dict_to_return.update({
                i: dict_f
            })

        return dict_to_return

    data_pred = get_data(y_pred, all_labels, False)
    data_true = get_data(y_true, all_labels, y_true_is_categorical)

    return tagging_metrics_from_dictionaries(data_pred, data_true, all_labels)


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

# EOF
