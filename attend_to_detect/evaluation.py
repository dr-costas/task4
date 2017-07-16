import numpy as np
import timeit
import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy, sigmoid
from sed_eval.sound_event import SegmentBasedMetrics

from attend_to_detect.dataset import (
    vehicle_classes, alarm_classes, get_input, get_output_binary)


MAX_VALIDATION_LEN = 5
EPS = 1e-5


def accuracy(output, target):
    # output : (batch, steps, classes)
    prediction = output.max(-1)[1].squeeze().type_as(target)

    # Don't pay for EOS
    prediction = prediction * (target != 0).type_as(prediction)

    numerator = torch.ne(prediction, target).type(torch.FloatTensor).sum()
    denominator = (target != 0).type_as(numerator).sum()
    numerator = numerator.data[0]
    denominator = denominator.data[0]
    if denominator < EPS:
        if numerator < EPS:
            error = 0.
        else:
            error = 1.
    else:
        error = numerator / denominator
    if error > 1.:
        error = 1
    return 100. * (1. - error)


def binary_accuracy(output, target):
    return ((sigmoid(output) >= 0.5).float() == target.float())\
        .float()\
        .mean(-2)\
        .mean()\
        .cpu()\
        .data\
        .numpy()[0]


def category_cost(out_hidden, target):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
    target_flat = target.view(-1)
    return cross_entropy(out_hidden_flat, target_flat)


def total_cost(hiddens, targets):
    return category_cost(hiddens[0], targets[0]) + \
           category_cost(hiddens[1], targets[1])


def binary_category_cost(out_hidden, target, weight=None):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
    target_flat = target.view(-1)
    if weight is not None:
        return binary_cross_entropy(sigmoid(out_hidden_flat), target_flat, weight=target.data + weight)
    return binary_cross_entropy(sigmoid(out_hidden_flat), target_flat)


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
        y_alarm_1_hot, y_alarm_logits = get_output_binary(batch[-2])

        # Get target values for vehicle classes
        y_vehicle_1_hot, y_vehicle_logits = get_output_binary(batch[-1])

        # Go through the common feature extractor
        common_features = common_feature_extractor(x)

        # Go through the alarm branch
        alarm_output, alarm_weights = branch_alarm(
            common_features, len(alarm_classes))

        # Go through the vehicle branch
        vehicle_output, vehicle_weights = branch_vehicle(
            common_features, len(vehicle_classes))

        # Calculate validation losses
        # Chopping of at the groundtruth length

        # alarm_output_aligned = alarm_output[:, :y_alarm_logits.size(1)].contiguous()
        # vehicle_output_aligned = vehicle_output[:, :y_vehicle_logits.size(1)].contiguous()

        loss_a += binary_category_cost(alarm_output, y_alarm_1_hot).data[0]
        loss_v += binary_category_cost(vehicle_output, y_vehicle_1_hot).data[0]

        # accuracy_a += accuracy(alarm_output_aligned, y_alarm_logits)
        # accuracy_v += accuracy(vehicle_output_aligned, y_vehicle_logits)
        accuracy_a += binary_accuracy(alarm_output, y_alarm_1_hot)
        accuracy_v += binary_accuracy(vehicle_output, y_vehicle_1_hot)

        valid_batches += 1

        if torch.has_cudnn:
            alarm_output = alarm_output.cpu()
            vehicle_output = vehicle_output.cpu()
            y_alarm_1_hot = y_alarm_1_hot.cpu()
            y_vehicle_1_hot = y_vehicle_1_hot.cpu()

        predictions_alarm.extend(sigmoid(alarm_output).data.numpy())
        predictions_vehicle.extend(sigmoid(vehicle_output).data.numpy())
        ground_truths_alarm.extend(y_alarm_1_hot.data.numpy())
        ground_truths_vehicle.extend(y_vehicle_1_hot.data.numpy())

    print('Epoch {:4d} validation elapsed time {:10.5f}'
          '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, validation_start_time - timeit.timeit(),
                loss_a/valid_batches, loss_v/valid_batches))
    metrics = tagging_metrics_from_raw_output(
        predictions_alarm, predictions_vehicle,
        ground_truths_alarm, ground_truths_vehicle,
        alarm_classes, vehicle_classes)
    print(metrics)

    # f_score_overall_alarm = metrics_alarm.overall_f_measure()
    # f_score_overall_vehicle = metrics_vehicle.overall_f_measure()
    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'valid_alarm': dict(
                        loss=loss_a/valid_batches,
                        acc=accuracy_a/valid_batches,
                        # f_score=f_score_overall_alarm['f_measure'],
                        # precision=f_score_overall_alarm['precision'],
                        # recall=f_score_overall_alarm['recall']
                    ),
                    'valid_vehicle': dict(
                        loss=loss_v/valid_batches,
                        acc=accuracy_v/valid_batches,
                        # f_score=f_score_overall_vehicle['f_measure'],
                        # precision=f_score_overall_vehicle['precision'],
                        # recall=f_score_overall_vehicle['recall']
                    )}})


def validate_separate_branches(valid_data, branch_alarm, branch_vehicle,
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
        y_alarm_1_hot, y_alarm_logits = get_output_binary(batch[-2])

        # Get target values for vehicle classes
        y_vehicle_1_hot, y_vehicle_logits = get_output_binary(batch[-1])

        # Go through the alarm branch
        alarm_output, alarm_weights = branch_alarm(
            x, len(alarm_classes))

        # Go through the vehicle branch
        vehicle_output, vehicle_weights = branch_vehicle(
            x, len(vehicle_classes))

        # Calculate validation losses
        # Chopping of at the groundtruth length

        # alarm_output_aligned = alarm_output[:, :y_alarm_logits.size(1)].contiguous()
        # vehicle_output_aligned = vehicle_output[:, :y_vehicle_logits.size(1)].contiguous()

        loss_a += binary_category_cost(alarm_output, y_alarm_1_hot).data[0]
        loss_v += binary_category_cost(vehicle_output, y_vehicle_1_hot).data[0]

        # accuracy_a += accuracy(alarm_output_aligned, y_alarm_logits)
        # accuracy_v += accuracy(vehicle_output_aligned, y_vehicle_logits)
        accuracy_a += binary_accuracy(alarm_output, y_alarm_1_hot)
        accuracy_v += binary_accuracy(vehicle_output, y_vehicle_1_hot)

        valid_batches += 1

        if torch.has_cudnn:
            alarm_output = alarm_output.cpu()
            vehicle_output = vehicle_output.cpu()
            y_alarm_1_hot = y_alarm_1_hot.cpu()
            y_vehicle_1_hot = y_vehicle_1_hot.cpu()

        predictions_alarm.extend(sigmoid(alarm_output).data.numpy())
        predictions_vehicle.extend(sigmoid(vehicle_output).data.numpy())
        ground_truths_alarm.extend(y_alarm_1_hot.data.numpy())
        ground_truths_vehicle.extend(y_vehicle_1_hot.data.numpy())

    print('Epoch {:4d} validation elapsed time {:10.5f}'
          '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, validation_start_time - timeit.timeit(),
                loss_a/valid_batches, loss_v/valid_batches))
    metrics = tagging_metrics_from_raw_output(
        predictions_alarm, predictions_vehicle,
        ground_truths_alarm, ground_truths_vehicle,
        alarm_classes, vehicle_classes)
    print(metrics)

    # f_score_overall_alarm = metrics_alarm.overall_f_measure()
    # f_score_overall_vehicle = metrics_vehicle.overall_f_measure()
    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'valid_alarm': dict(
                        loss=loss_a/valid_batches,
                        acc=accuracy_a/valid_batches,
                        # f_score=f_score_overall_alarm['f_measure'],
                        # precision=f_score_overall_alarm['precision'],
                        # recall=f_score_overall_alarm['recall']
                    ),
                    'valid_vehicle': dict(
                        loss=loss_v/valid_batches,
                        acc=accuracy_v/valid_batches,
                        # f_score=f_score_overall_vehicle['f_measure'],
                        # precision=f_score_overall_vehicle['precision'],
                        # recall=f_score_overall_vehicle['recall']
                    )}})


def validate_single_branch(valid_data, branch_alarm, branch_vehicle,
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
        y_alarm_1_hot, y_alarm_logits = get_output_binary(batch[-2])

        # Get target values for vehicle classes
        y_vehicle_1_hot, y_vehicle_logits = get_output_binary(batch[-1])

        # Go through the alarm branch
        alarm_output, alarm_weights = branch_alarm(
            x, len(alarm_classes))

        # Go through the vehicle branch
        vehicle_output, vehicle_weights = branch_vehicle(
            x, len(vehicle_classes))

        # Calculate validation losses
        # Chopping of at the groundtruth length

        # alarm_output_aligned = alarm_output[:, :y_alarm_logits.size(1)].contiguous()
        # vehicle_output_aligned = vehicle_output[:, :y_vehicle_logits.size(1)].contiguous()

        loss_a += binary_category_cost(alarm_output, y_alarm_1_hot).data[0]
        loss_v += binary_category_cost(vehicle_output, y_vehicle_1_hot).data[0]

        # accuracy_a += accuracy(alarm_output_aligned, y_alarm_logits)
        # accuracy_v += accuracy(vehicle_output_aligned, y_vehicle_logits)
        accuracy_a += binary_accuracy(alarm_output, y_alarm_1_hot)
        accuracy_v += binary_accuracy(vehicle_output, y_vehicle_1_hot)

        valid_batches += 1

        if torch.has_cudnn:
            alarm_output = alarm_output.cpu()
            vehicle_output = vehicle_output.cpu()
            y_alarm_1_hot = y_alarm_1_hot.cpu()
            y_vehicle_1_hot = y_vehicle_1_hot.cpu()

        predictions_alarm.extend(sigmoid(alarm_output).data.numpy())
        predictions_vehicle.extend(sigmoid(vehicle_output).data.numpy())
        ground_truths_alarm.extend(y_alarm_1_hot.data.numpy())
        ground_truths_vehicle.extend(y_vehicle_1_hot.data.numpy())

    print('Epoch {:4d} validation elapsed time {:10.5f}'
          '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, validation_start_time - timeit.timeit(),
                loss_a/valid_batches, loss_v/valid_batches))
    metrics = tagging_metrics_from_raw_output(
        predictions_alarm, predictions_vehicle,
        ground_truths_alarm, ground_truths_vehicle,
        alarm_classes, vehicle_classes)
    print(metrics)

    # f_score_overall_alarm = metrics_alarm.overall_f_measure()
    # f_score_overall_vehicle = metrics_vehicle.overall_f_measure()
    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'valid_alarm': dict(
                        loss=loss_a/valid_batches,
                        acc=accuracy_a/valid_batches,
                        # f_score=f_score_overall_alarm['f_measure'],
                        # precision=f_score_overall_alarm['precision'],
                        # recall=f_score_overall_alarm['recall']
                    ),
                    'valid_vehicle': dict(
                        loss=loss_v/valid_batches,
                        acc=accuracy_v/valid_batches,
                        # f_score=f_score_overall_vehicle['f_measure'],
                        # precision=f_score_overall_vehicle['precision'],
                        # recall=f_score_overall_vehicle['recall']
                    )}})


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


def tagging_metrics_from_raw_output(y_pred_a, y_pred_v, y_true_a, y_true_v, all_labels_a, all_labels_v):
    """

    If 1-hot-encoding, the value [1, 0, 0, ..., 0] is considered <EOS>. \
    If not 1-hot-encoded, the value 0 is considered <EOS>.

    :param y_pred_a: List with predictions, of len = nb_files and each element \
                   is a matrix of size [nb_events_detected x nb_labels]
    :type y_pred_a: numpy.core.multiarray.ndarray
    :param y_true_a: List with true values, of len = nb_files. If 1-hot-enconding \
                   each element is a matrix of size [nb_events x nb_labels] . \
                   Else, each element is an array of len = nb_events.
    :type y_true_a: numpy.core.multiarray.ndarray
    :param all_labels_a: A list with all the labels **including** the <EOS> at the 0th index
    :type all_labels_a: list[str]
    :param y_pred_v: List with predictions, of len = nb_files and each element \
                   is a matrix of size [nb_events_detected x nb_labels]
    :type y_pred_v: numpy.core.multiarray.ndarray
    :param y_true_v: List with true values, of len = nb_files. If 1-hot-enconding \
                   each element is a matrix of size [nb_events x nb_labels] . \
                   Else, each element is an array of len = nb_events.
    :type y_true_v: numpy.core.multiarray.ndarray
    :param all_labels_v: A list with all the labels **including** the <EOS> at the 0th index
    :type all_labels_v: list[str]
    :return:
    :rtype:
    """
    def get_data(the_data, the_labels):
        all_files_list = []
        rounded_data = np.round(the_data)

        for file_data in rounded_data:
            file_list = []
            for i, class_data in enumerate(file_data):
                if class_data == 1:
                    file_list.append(
                        {
                            'event_offset': 10.00,
                            'event_onset': 00.00,
                            'event_label': the_labels[i]
                        }
                    )
            all_files_list.append(file_list)

        return all_files_list

    data_pred_a = get_data(y_pred_a, all_labels_a)
    data_true_a = get_data(y_true_a, all_labels_a)

    data_pred_v = get_data(y_pred_v, all_labels_v)
    data_true_v = get_data(y_true_v, all_labels_v)

    data_pred = []
    for f_data_a, f_data_v in zip(data_pred_a, data_pred_v):
        data_pred.append(f_data_a + f_data_v)

    data_true = []
    for f_data_a, f_data_v in zip(data_true_a, data_true_v):
        data_true.append(f_data_a + f_data_v)

    return tagging_metrics_from_list(data_pred, data_true, all_labels_a + all_labels_v)


def main():
    # Test code
    pass


if __name__ == '__main__':
    main()
