#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sed_eval.sound_event import SegmentBasedMetrics


__docformat__ = 'reStructuredText'


def tagging_metrics_from_list(y_pred_dict, y_true_dict, all_labels):
    """

    :param y_pred_dict:
    :type y_pred_dict: list[list[dict[str, list[str|float]]]]
    :param y_true_dict:
    :type y_true_dict: list[list[dict[str, list[str|float]]]]
    :param all_labels:
    :type all_labels: list[str]
    :return: The metrics
    :rtype:
    """

    segment_based_metrics = SegmentBasedMetrics(event_label_list=all_labels,
                                                time_resolution=1)

    for pred_data, true_data in zip(y_pred_dict, y_true_dict):
        segment_based_metrics.evaluate(true_data, pred_data)

    # Or print all metrics as reports
    return segment_based_metrics


def tagging_metrics_from_raw_output(y_pred, y_true, all_labels, y_true_is_categorical=True):
    """

    If 1-hot-encoding, the value [1, 0, 0, ..., 0] is considered <EOS>. \
    If not 1-hot-encoded, the value 0 is considered <EOS>.

    :param y_pred: List with predictions, of len = nb_files and each element \
                   is a matrix of size [nb_events_detected x nb_labels]
    :type y_pred: numpy.core.multiarray.ndarray
    :param y_true: List with true values, of len = nb_files. If 1-hot-enconding \
                   each element is a matrix of size [nb_events x nb_labels] . \
                   Else, each element is an array of len = nb_events.
    :type y_true: numpy.core.multiarray.ndarray
    :param all_labels: A list with all the labels **including** the <EOS> at the 0th index
    :type all_labels: list[str]
    :param y_true_is_categorical: Indication if y_true is categorical
    :type y_true_is_categorical: bool
    :return:
    :rtype:
    """
    def get_data(the_data, the_labels, is_categorical):
        all_files_list = []

        for i, file_data in enumerate(the_data):
            found_eos = False
            file_list = []
            for event in file_data:
                if is_categorical:
                    index = event
                else:
                    index = np.argmax(event)

                if index == 0 and not found_eos:
                    found_eos = True
                elif index == 0 and found_eos:
                    break

                file_list.append(
                    {
                        'event_offset': 10.00,
                        'event_onset': 00.00,
                        'event_label': the_labels[int(index)]
                    }
                )
            all_files_list.append(file_list)

        return all_files_list

    data_pred = get_data(y_pred, all_labels, False)
    data_true = get_data(y_true, all_labels, y_true_is_categorical)

    return tagging_metrics_from_list(data_pred, data_true, all_labels)


def main():
    # Test code
    labels = ['eos', 'car', 'bus', 'truck', 'bike']
    nb_files = 10

    x = np.random.random((nb_files, 3, len(labels)))
    y_1_hot = np.zeros((nb_files, 3, len(labels)))
    for i in range(nb_files):
        random_col = np.random.choice(len(labels), 3, replace=False)
        for ii, v in enumerate(random_col):
            y_1_hot[i, ii, v] = 1

    y_categorical = np.zeros(y_1_hot.shape[:-1])

    for i, e in enumerate(y_1_hot):

        events_ids = []

        for ee in e:
            events_ids.append(np.argmax(ee).astype('int'))

        y_categorical[i, :] = events_ids

    print(tagging_metrics_from_raw_output(x, y_categorical, labels, y_true_is_categorical=True))
    print(tagging_metrics_from_raw_output(x, y_1_hot, labels, y_true_is_categorical=False))


if __name__ == '__main__':
    main()

# EOF
