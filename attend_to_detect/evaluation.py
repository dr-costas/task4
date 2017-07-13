import numpy as np
import timeit
import torch
from torch.nn.functional import cross_entropy, softmax
from sed_eval.sound_event import SegmentBasedMetrics

from attend_to_detect.dataset import (
    vehicle_classes, alarm_classes, get_input, get_output)


def accuracy(output, target):
    acc = (100. * torch.eq(output.max(2)[1].squeeze().type_as(target), target).type(torch.FloatTensor)).mean()
    return acc.data[0]


def category_cost(out_hidden, target):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
    target_flat = target.view(-1)
    return cross_entropy(out_hidden_flat, target_flat)


def total_cost(hiddens, targets):
    return category_cost(hiddens[0], targets[0]) + \
           category_cost(hiddens[1], targets[1])


def validate(valid_data, common_feature_extractor, branch_alarm, branch_vehicle,
             scaler, logger, total_iterations, epoch):
    valid_batches = 0
    loss_a = 0.0
    loss_v = 0.0

    accuracy_a = 0.0
    accuracy_v = 0.0

    validation_start_time = timeit.timeit()
    predictions_alarm = []
    predictions_vehicle = []
    ground_truths_alarm = []
    ground_truths_vehicle = []
    for batch in valid_data.get_epoch_iterator():
        # Get input
        x = get_input(batch[0], scaler, volatile=True)

        # Get target values for alarm classes
        y_alarm_1_hot, y_alarm_logits = get_output(batch[-2])

        # Get target values for vehicle classes
        y_vehicle_1_hot, y_vehicle_logits = get_output(batch[-1])

        # Go through the common feature extractor
        common_features = common_feature_extractor(x)

        # Go through the alarm branch
        alarm_output, alarm_weights = branch_alarm(common_features, y_alarm_logits.size(1))

        # Go through the vehicle branch
        vehicle_output, vehicle_weights = branch_vehicle(common_features, y_vehicle_logits.size(1))

        # Calculate validation losses
        loss_a += category_cost(alarm_output, y_alarm_logits).data[0]
        loss_v += category_cost(vehicle_output, y_vehicle_logits).data[0]

        accuracy_a += accuracy(alarm_output, y_alarm_logits)
        accuracy_v += accuracy(vehicle_output, y_vehicle_logits)

        valid_batches += 1

        if torch.has_cudnn:
            alarm_output = alarm_output.cpu()
            vehicle_output = vehicle_output.cpu()
            y_alarm_logits = y_alarm_logits.cpu()
            y_vehicle_logits = y_vehicle_logits.cpu()

        predictions_alarm.extend(softmax(alarm_output).data.numpy())
        predictions_vehicle.extend(softmax(vehicle_output).data.numpy())
        ground_truths_alarm.extend(y_alarm_logits.data.numpy())
        ground_truths_vehicle.extend(y_vehicle_logits.data.numpy())

    print('Epoch {:4d} validation elapsed time {:10.5f}'
          '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, validation_start_time - timeit.timeit(),
                loss_a/valid_batches, loss_v/valid_batches))
    metrics_alarm = tagging_metrics_from_raw_output(
        predictions_alarm, ground_truths_alarm, ['<EOS>'] + alarm_classes)
    metrics_vehicle = tagging_metrics_from_raw_output(
        predictions_vehicle, ground_truths_vehicle, ['<EOS>'] + vehicle_classes)
    print(metrics_alarm)
    print(metrics_vehicle)

    f_score_overall_alarm = metrics_alarm.overall_f_measure()
    f_score_overall_vehicle = metrics_vehicle.overall_f_measure()
    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'valid_alarm': dict(
                        loss=loss_a/valid_batches,
                        acc=accuracy_a/valid_batches,
                        f_score=f_score_overall_alarm['f_measure'],
                        precision=f_score_overall_alarm['precision'],
                        recall=f_score_overall_alarm['recall']),
                    'valid_vehicle': dict(
                        loss=loss_v/valid_batches,
                        acc=accuracy_v/valid_batches,
                        f_score=f_score_overall_vehicle['f_measure'],
                        precision=f_score_overall_vehicle['precision'],
                        recall=f_score_overall_vehicle['recall'])}})


def tagging_metrics_from_dictionaries(y_pred_dict, y_true_dict, all_labels, file_prefix=''):
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
            file_list = []
            for event in file_data:
                if is_categorical:
                    index = event
                else:
                    index = np.argmax(event)

                if index == 0:
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

    return tagging_metrics_from_list(data_pred, data_true, all_labels[1:])


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

    x = tagging_metrics_from_raw_output(x, y_categorical, labels, y_true_is_categorical=True)
    print('OK')
    print(tagging_metrics_from_raw_output(x, y_1_hot, labels, y_true_is_categorical=False))


if __name__ == '__main__':
    main()
