import numpy as np
from sed_eval.sound_event import SegmentBasedMetrics


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

    segment_based_metrics = SegmentBasedMetrics(event_label_list=all_labels,
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
