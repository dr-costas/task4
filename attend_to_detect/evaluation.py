import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.nn.functional import cross_entropy, binary_cross_entropy, sigmoid
from sed_eval.sound_event import SegmentBasedMetrics
from attend_to_detect.evaluation_aux import FileFormat

from attend_to_detect.dataset import (
    vehicle_classes, alarm_classes, get_input, get_output_binary, get_output_binary_single,
    get_output_binary_one_hot, get_output_one_hot, get_output_new_model)

from attend_to_detect.utils.msa_utils import *


MAX_VALIDATION_LEN = 5
EPS = 1e-5

class_freqs_alarm = [383., 341., 192., 260., 574., 2557., 2491., 1533., 695.]

class_freqs_vehicle = [2073., 1646., 27218., 3882., 3962., 7492., 3426., 2256.]

all_freqs_alarms_first = class_freqs_alarm + class_freqs_vehicle
all_freqs_vehicles_first = class_freqs_vehicle + class_freqs_alarm


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
        .float() / np.array(all_freqs_alarms_first)\
        .mean(-2)\
        .mean()\
        .cpu()\
        .data\
        .numpy()[0]


def binary_accuracy_single(output, target, is_valid=False):
    output = output.sum(1).view(output.size(0), output.size(2))[:, 1:]
    target = target[:, 1:]
    acc = ((sigmoid(output) >= 0.5).float() == target.float()).float()
    if not is_valid:
        weights = np.array(all_freqs_alarms_first).reshape(1, len(all_freqs_alarms_first))
        weights = torch.autograd.Variable(torch.from_numpy(weights).float())
        weights = weights.type_as(acc)
        acc = acc / weights.expand_as(acc)
    acc = acc.mean(-2).cpu().data.numpy()[0]

    return acc


def category_cost(out_hidden, target):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
    target_flat = target.view(-1)
    return cross_entropy(out_hidden_flat, target_flat)


def total_cost(hiddens, targets):
    return category_cost(hiddens[0], targets[0]) + \
           category_cost(hiddens[1], targets[1])


def binary_category_cost(out_hidden, target, weight=None):
    out_hidden_flat = out_hidden.view(-1, out_hidden.size(1))
    target_flat = target.view(-1)
    if weight is not None:
        return binary_cross_entropy(sigmoid(out_hidden_flat), target_flat, weight=target.data + weight)
    return binary_cross_entropy(sigmoid(out_hidden_flat), target_flat)


def binary_accuracy_single_multi_label_approach(output, target):

    out = torch.zeros(output.size()).float()
    _, indices = torch.max(output.data, -1)

    for i, index in enumerate(indices):
        for i2, index2 in enumerate(index):
            out[i, i2, index2[0]] = 1.0

    out = torch.autograd.Variable(out.sum(1).squeeze())
    if torch.has_cudnn:
        out = out.cuda()

    acc = (out == target.float()).float()
    acc = (acc.sum(-1) == acc.size()[-1]).float().mean()

    if torch.has_cudnn:
        acc = acc.cpu()

    return acc.data[0]


def accuracy_single_one_hot(output, target):

    acc = 0

    for i in range(output.size()[1]):
        s = torch.nn.functional.softmax(output[:, i, :])
        _, m = s.max(-1)
        s = torch.autograd.Variable(torch.zeros(m.size()))
        if torch.has_cudnn:
            s = s.cuda()
        for ii, e in enumerate(m):
            s[ii] = e

        acc += (s.squeeze() == target[:, i]).float().sum()/target.size()[0]

    return (acc/target.size()[1]).data[0]


def flatten(x):
    return x.view(x.size(0) * x.size(1), -1)


def deflatten(x, shape):
    return x.view(shape[0], shape[1], x.size(1))


def per_example_cross_entropy(x, targets):
    flat_x = flatten(x)
    flat_logits = torch.nn.functional.sigmoid(flat_x)
    logits = deflatten(flat_logits, x.size())
    targets = deflatten(targets.view(-1, 1), x.size())
    loss = -torch.gather(logits, 2, targets)
    loss = loss.mean(dim=0)
    return loss


def manual_b_entropy(pred, true):
    weights = np.array(all_freqs_alarms_first).reshape(1, len(all_freqs_alarms_first), 1)
    weights = torch.autograd.Variable(torch.from_numpy(50000 / weights).float())
    if torch.has_cudnn:
        weights = weights.cuda()
    local_pred = pred.view(pred.size()[:-1])
    local_true = true.view(true.size()[:-1])
    local_weights = weights.view(weights.size()[:-1]).expand_as(local_true)

    s = sigmoid(local_pred)

    r = local_weights * (local_true*torch.log(s)) + \
        (1 - local_true) * torch.log(1 - s)

    # r = torch.autograd.Variable(torch.zeros(pred.size()).float())
    #
    # for i, (t, o, w) in enumerate(zip(local_true, local_pred, local_weights)):
    #     r[i, :] = w * (t*torch.log(o)) + (1 - t) * torch.log(1 - o)

    return (-r.sum(0)).mean()


def binary_category_cost_single(out_hidden, target, weight=None, is_valid=False):
    weights = np.array(all_freqs_alarms_first).reshape(1, len(all_freqs_alarms_first), 1)
    weights = torch.autograd.Variable(torch.from_numpy(1.0/weights).float())
    if torch.has_cudnn:
        weights = weights.cuda()
    a_term = target * weights.expand_as(target)
    b_term = (1-target) * (1-weights.expand_as(target))
    weight = a_term + b_term
    local_pred = out_hidden.view(out_hidden.size()[:-1])
    local_true = target.view(target.size()[:-1])
    local_weights = weight.view(weight.size()[:-1])

    r = torch.autograd.Variable(torch.zeros(out_hidden.size()).float())

    for i, (t, o, w) in enumerate(zip(local_true, local_pred, local_weights)):
        r[i, :] = w * (t*torch.log(o + EPS)) + (1 - t) * torch.log(1 - o + EPS) + EPS

    return r.mean()

    # return torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(out_hidden), target,
    #                                                 weight=weight[0, :, :])


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits:
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.
    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    if weight is not None and target.dim() != 1:
        weight = weight.view(1, target.size(1)).expand_as(target)

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()

    if weight is not None:
        loss = loss * weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def multi_label_loss(y_pred, y_true, use_weights):
    out_hidden_summed = y_pred.sum(1).squeeze()
    if use_weights:
        local_weights = [30000.0 / a for a in all_freqs_alarms_first]
        weights = np.array([1.] + local_weights).reshape(1, len(all_freqs_alarms_first) + 1, 1)
        weights = torch.autograd.Variable(torch.from_numpy(weights).float())
        if torch.has_cudnn:
            weights = weights.cuda()
        return binary_cross_entropy_with_logits(out_hidden_summed, y_true, weights)
    else:
        return binary_cross_entropy_with_logits(out_hidden_summed, y_true)


def loss_one_hot_single(y_pred, y_true, use_weights):
    if use_weights:
        local_weights_positive = [50000.0 / a for a in all_freqs_vehicles_first]
        local_weights_negative = [50000.0 / (50000.0 - a) for a in all_freqs_vehicles_first]
        weights = torch.ones(y_pred.size()[-1]).float()
        local_weights_positive = torch.from_numpy(np.array(local_weights_positive))
        local_weights_negative = torch.from_numpy(np.array(local_weights_negative))
        weights[1::2] = local_weights_positive
        weights[0::2] = local_weights_negative
        if torch.has_cudnn:
            weights = weights.cuda()
    else:
        weights = None
    return torch.nn.functional.cross_entropy(
        y_pred.view(y_pred.size()[0] * y_pred.size()[1], y_pred.size()[-1]),
        y_true.view(y_true.size()[0] * y_true.size()[1]).long(),
        weight=weights
    )


def loss_new_model(y_pred, y_true, use_weights, total_examples, weight_factor):

    def loss_positive(y_pred_inner, the_class_weight):
        target_val = Variable(torch.ones((1,)))
        if torch.has_cudnn:
            target_val = target_val.cuda()
        return torch.nn.functional.binary_cross_entropy(
            y_pred_inner, target_val,
            weight=the_class_weight, size_average=False
        )

    def loss_negative(y_pred_inner, the_class_weight):
        target_val = Variable(torch.zeros((1,)))
        if torch.has_cudnn:
            target_val = target_val.cuda()
        return torch.nn.functional.binary_cross_entropy(
            y_pred_inner, target_val,
            weight=the_class_weight, size_average=False
        )

    if use_weights:
        local_weights_positive = [weight_factor / a for a in all_freqs_vehicles_first]
        weights = torch.from_numpy(np.array(local_weights_positive)).float()
    else:
        weights = None

    if torch.has_cudnn and weights is not None:
        weights = weights.cuda()

    loss = Variable(torch.zeros((1, )), requires_grad=True)

    if torch.has_cudnn:
        loss = loss.cuda()

    total_additions = 0

    for b in range(y_pred.size()[0]):
        for c_target in range(y_true.size()[1]):
            target_class = y_true[b, c_target].data[0]
            if target_class > -1:
                for c in range(y_pred.size()[1]):
                    w = weights[target_class:target_class+1]
                    y = y_pred[b, c]
                    if c != target_class:
                        loss += loss_negative(y, w)
                    else:
                        loss += loss_positive(y, w)
                    total_additions += 1

    return loss/total_examples
    # return loss/total_additions


def validate(valid_data, common_feature_extractor, branch_alarm, branch_vehicle,
             scaler, logger, total_iterations, epoch):
    valid_batches = 0
    loss_a = 0.0
    loss_v = 0.0

    accuracy_a = 0.0
    accuracy_v = 0.0

    validation_start_time = time.time()
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

    print('Epoch {:4d} validation elapsed time {:10.5f} sec(s)'
          '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, time.time() - validation_start_time,
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

    validation_start_time = time.time()
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

        loss_a += binary_category_cost(alarm_output, y_alarm_1_hot).data[0]
        loss_v += binary_category_cost(vehicle_output, y_vehicle_1_hot).data[0]

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

    print('Epoch {:4d} validation elapsed time {:10.5f} sec(s)'
          '\n\tValid. loss alarm: {:10.6f} | vehicle: {:10.6f} '.format(
                epoch, time.time() - validation_start_time,
                loss_a/valid_batches, loss_v/valid_batches))
    metrics = tagging_metrics_from_raw_output(
        predictions_alarm, predictions_vehicle,
        ground_truths_alarm, ground_truths_vehicle,
        alarm_classes, vehicle_classes)
    print(metrics)

    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'valid_alarm': dict(
                        loss=loss_a/valid_batches,
                        acc=accuracy_a/valid_batches
                    ),
                    'valid_vehicle': dict(
                        loss=loss_v/valid_batches,
                        acc=accuracy_v/valid_batches
                    )}})


def validate_single_branch(valid_data, network, scaler, logger, total_iterations, epoch):
    valid_batches = 0
    loss = 0.0
    accuracy = 0.0

    validation_start_time = time.time()
    predictions = []
    ground_truths = []
    # loss_fn = torch.nn.MSELoss()
    for batch in valid_data.get_epoch_iterator():
        # Get input
        x = get_input(batch[0], scaler, volatile=True)

        # Get target values for alarm classes
        y_1_hot = get_output_binary_single(batch[-2], batch[-1])

        # Go through the alarm branch
        output, attention_weights = network(x, y_1_hot.shape[1])

        loss += manual_b_entropy(output, y_1_hot).data[0]
        accuracy += binary_accuracy_single(output, y_1_hot, is_valid=True)

        valid_batches += 1

        if torch.has_cudnn:
            output = output.cpu()
            y_1_hot = y_1_hot.cpu()

        predictions.extend(sigmoid(output).data.numpy())
        ground_truths.extend(y_1_hot.data.numpy())

    print('Epoch {:4d} validation elapsed time {:10.5f} sec(s) | Valid. loss alarm: {:10.6f}'.format(
                epoch, time.time() - validation_start_time,
                loss/valid_batches))
    metrics = tagging_metrics_from_raw_output_single(
        predictions, ground_truths, alarm_classes + vehicle_classes)
    print(metrics)

    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'validation': dict(
                        loss=loss/valid_batches,
                        acc=accuracy/valid_batches
                    )
                }})


def validate_single_branch_multi_label_approach(valid_data, network, scaler, logger, total_iterations,
                                                epoch, use_weights):
    valid_batches = 0
    loss = 0.0
    accuracy = 0.0

    validation_start_time = time.time()
    predictions = []
    ground_truths = []
    # loss_fn = torch.nn.MSELoss()
    for batch in valid_data.get_epoch_iterator():
        # Get input
        x = get_input(batch[0], scaler, volatile=True)

        # Get target values for alarm classes
        y_1_hot, y_categorical = get_output_binary_one_hot(batch[-2], batch[-1])

        # Go through the alarm branch
        output, attention_weights = network(x[:, :, :, :64], y_1_hot.shape[1])
        target_values = torch.autograd.Variable(torch.from_numpy(y_1_hot.sum(axis=1)).float())
        if torch.has_cudnn:
            target_values = target_values.cuda()

        loss += multi_label_loss(torch.nn.functional.softmax(output), target_values, use_weights).data[0]
        accuracy += binary_accuracy_single_multi_label_approach(output, target_values)

        valid_batches += 1

        if torch.has_cudnn:
            output = output.cpu()

        predictions.extend(torch.nn.functional.softmax(output).data.numpy())
        ground_truths.extend(y_1_hot)

    print('Epoch {:3d} | Elapsed valid. time {:10.3f} sec(s) | Valid. loss: {:10.6f}'.format(
                epoch, time.time() - validation_start_time,
                loss/valid_batches))
    metrics = tagging_metrics_from_raw_output_single_multi_label(
        predictions, ground_truths, alarm_classes + vehicle_classes)
    print(metrics)

    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'validation': dict(
                        loss=loss/valid_batches,
                        acc=accuracy/valid_batches
                    )
                }})


def validate_single_one_hot(valid_data, network, scaler, logger, total_iterations,
                                                epoch, use_weights):
    valid_batches = 0
    loss = 0.0

    validation_start_time = time.time()
    predictions = []
    ground_truths = []
    # loss_fn = torch.nn.MSELoss()
    for batch in valid_data.get_epoch_iterator():
        # Get input
        x = get_input(batch[0], scaler, volatile=True)

        # Get target values for alarm classes
        y_1_hot, y_categorical = get_output_one_hot(batch[-2], batch[-1])

        # Go through the alarm branch
        # output, attention_weights = network(x, y_1_hot.size()[1])
        output, attention_weights = network(x[:, :, :, :64], y_1_hot.size()[1])

        loss += loss_one_hot_single(output, y_categorical, use_weights).data[0]
        # accuracy += accuracy_single_one_hot(output, y_categorical)

        valid_batches += 1

        for i in range(len(output)):
            output[i, :, :] = torch.nn.functional.softmax(output[i, :, :])

        if torch.has_cudnn:
            output = output.cpu()
            y_1_hot = y_1_hot.cpu()

        y_1_hot = y_1_hot.data.numpy()

        output = output.data.numpy()
        arg_max = np.argmax(output, axis=-1)

        for i in range(len(output)):
            for ii in range(len(output[i])):
                output[i, ii, :] = 0
                output[i, ii, arg_max[i, ii]] = 1

        predictions.extend(output[:, :, 1::2])
        ground_truths.extend(y_1_hot[:, :, 1::2])

    print('Epoch {:3d} | Elapsed valid. time {:10.3f} sec(s) | Valid. loss: {:10.6f}'.format(
                epoch, time.time() - validation_start_time,
                loss/valid_batches))
    metrics = tagging_metrics_one_hot(
        predictions, ground_truths,  vehicle_classes + alarm_classes, True)
    # print(metrics)

    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'validation': dict(
                        loss=loss/valid_batches,
                        acc=metrics['f1']
                    )
                }})


def validate_single_new_model(valid_data, network, scaler, logger, total_iterations,
                              epoch, config, s, total_validation_examples):
    valid_batches = 0
    loss = 0.0

    validation_start_time = time.time()
    predictions = []
    ground_truths = []
    # loss_fn = torch.nn.MSELoss()
    for batch in valid_data.get_epoch_iterator():
        # Get input
        x = get_input(batch[0], scaler, volatile=True)

        # Get target values for alarm classes
        y_1_hot, y_categorical = get_output_new_model(batch[-2], batch[-1])

        # Go through the alarm branch
        output, mlp_output = network(x[:, :, :, :config.nb_features])

        if torch.has_cudnn:
            tmp_v = output.cpu().data.numpy()
            tmp_classes = y_categorical.cpu().data.numpy()
        else:
            tmp_v = output.data.numpy()
            tmp_classes = y_categorical.data.numpy()

        #max_means, s_i_inds, max_means_index_end, max_means_index_end = find_max_mean(
        #    tmp_v, tmp_classes, s, len(vehicle_classes) + len(alarm_classes))

        # mult_result = torch.autograd.Variable(torch.zeros(output.size()))

        # if torch.has_cudnn:
        #    mult_result = mult_result.cuda()

        # for b_i in range(s_i_inds.shape[0]):
        #    for c_i in range(s_i_inds.shape[1]):
        #        if s_i_inds[b_i, c_i] == -1:
        #            mult_result[b_i, :, c_i] = -1 * output[b_i, :, c_i]
        #        else:
        #            s_tmp = torch.autograd.Variable(torch.from_numpy(
        #                s[int(s_i_inds[b_i, c_i]), :]
        #            ).float())
        #            if torch.has_cudnn:
        #                s_tmp = s_tmp.cuda()
        #            mult_result[b_i, :, c_i] = output[b_i, :, c_i] * s_tmp

        final_output = torch.nn.functional.sigmoid(mlp_output * output.mean(1))

        # Calculate losses, do backward passing, and do updates
        loss_tmp = loss_new_model(
            final_output, y_categorical,
            config.network_loss_weight,
            config.batch_size, config.weighting_factor)

        if torch.has_cudnn:
            loss_tmp = loss_tmp.cpu()

        loss += loss_tmp.data[0]

        valid_batches += 1

        if torch.has_cudnn:
            final_output = final_output.cpu()
            y_categorical = y_categorical.cpu()

        y_categorical = y_categorical.data.numpy()
        final_output = final_output.data.numpy()
        predictions.extend(final_output)
        ground_truths.extend(y_categorical)

    print('Epoch {:3d} | Elapsed valid. time {:10.3f} sec(s) | Valid. loss: {:10.6f}'.format(
                epoch, time.time() - validation_start_time,
                loss/valid_batches))
    metrics = tagging_metrics_categorical(
        predictions, ground_truths,  vehicle_classes + alarm_classes, True)

    logger.log({'iteration': total_iterations,
                'epoch': epoch,
                'records': {
                    'validation': dict(
                        loss=loss/valid_batches,
                        acc=metrics['f1']
                    )
                }})


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

        for i_f, file_data in enumerate(rounded_data):
            file_list = []
            for i, class_data in enumerate(file_data):
                if class_data == 1:
                    file_list.append(
                        {
                            'audio_file': 'audio_file_id'.format(i_f),
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


def tagging_metrics_from_raw_output_single(y_pred, y_true, all_labels):
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
    :return:
    :rtype:
    """
    def get_data(the_data, the_labels):
        all_files_list = []
        rounded_data = np.round(the_data)

        for i_f, file_data in enumerate(rounded_data):
            file_list = []
            for i, class_data in enumerate(file_data):
                if class_data == 1:
                    file_list.append(
                        {
                            'audio_file': 'audio_file_id'.format(i_f),
                            'event_offset': 10.00,
                            'event_onset': 00.00,
                            'event_label': the_labels[i]
                        }
                    )
            all_files_list.append(file_list)

        return all_files_list

    data_pred = []
    for f_data_a in get_data(y_pred, all_labels):
        data_pred.append(f_data_a)

    data_true = []
    for f_data_a in get_data(y_true, all_labels):
        data_true.append(f_data_a)

    return tagging_metrics_from_list(data_pred, data_true, all_labels)


def tagging_metrics_from_raw_output_single_multi_label(y_pred, y_true, all_labels):
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
    :return:
    :rtype:
    """
    def get_data(the_data, the_labels):
        all_files_list = []

        for i_f, file_data in enumerate(the_data):
            file_list = []
            for i in np.argmax(file_data, axis=-1):
                if i != 0:
                    file_list.append(
                        {
                            'audio_file': 'audio_file_id'.format(i_f),
                            'event_offset': 10.00,
                            'event_onset': 00.00,
                            'event_label': the_labels[i-1]
                        }
                    )
            all_files_list.append(file_list)

        return all_files_list

    data_pred = []
    for f_data_a in get_data(y_pred, all_labels):
        data_pred.append(f_data_a)

    data_true = []
    for f_data_a in get_data(y_true, all_labels):
        data_true.append(f_data_a)

    return tagging_metrics_from_list(data_pred, data_true, all_labels)


def tagging_metrics_one_hot(y_pred, y_true, all_labels, print_out=False):
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
    :return:
    :rtype:
    """
    def get_data(the_data, the_labels):
        all_files_list = []

        for i_f, file_data in enumerate(the_data):
            file_list = []
            for i in np.nonzero(np.sum(file_data, axis=-2))[0]:
                file_list.append(
                    {
                        'audio_file': 'audio_file_id_{}'.format(i_f),
                        'event_offset': 10.00,
                        'event_onset': 00.00,
                        'event_label': the_labels[i]
                    }
                )
            all_files_list.append(file_list)

        return all_files_list

    data_pred = []
    for f_data_a in get_data(y_pred, all_labels):
        data_pred.append(f_data_a)

    data_true = []
    for f_data_a in get_data(y_true, all_labels):
        data_true.append(f_data_a)

    return task_a_evaluation(data_pred, data_true, print_out)
    # return tagging_metrics_from_list(data_pred, data_true, all_labels)


def tagging_metrics_categorical(y_pred, y_true, all_labels, print_out=False):
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
    :param print_out:
    :type print_out: bool
    :return:
    :rtype:
    """
    def get_data_pred(the_data, the_labels):
        all_files_list = []

        for i_f, file_data in enumerate(the_data):
            file_list = []
            for i_d, i in enumerate(file_data):
                if i > 0.5:
                    file_list.append(
                        {
                            'audio_file': 'audio_file_id_{}'.format(i_f),
                            'event_offset': 10.00,
                            'event_onset': 00.00,
                            'event_label': the_labels[i_d]
                        }
                    )
            if len(file_list) == 0:
                file_list.append(
                    {
                        'audio_file': 'audio_file_id_{}'.format(i_f),
                        'event_offset': 10.00,
                        'event_onset': 00.00,
                        'event_label': ''
                    }
                )
            all_files_list.append(file_list)

        return all_files_list

    def get_data_true(the_data, the_labels):
        all_files_list = []

        for i_f, file_data in enumerate(the_data):
            file_list = []
            for i in file_data:
                if i > -1:
                    file_list.append(
                        {
                            'audio_file': 'audio_file_id_{}'.format(i_f),
                            'event_offset': 10.00,
                            'event_onset': 00.00,
                            'event_label': the_labels[i]
                        }
                    )
            all_files_list.append(file_list)

        return all_files_list

    data_pred = []
    for f_data_a in get_data_pred(y_pred, all_labels):
        data_pred.append(f_data_a)

    data_true = []
    for f_data_a in get_data_true(y_true, all_labels):
        data_true.append(f_data_a)

    return task_a_evaluation(data_pred, data_true, print_out)


def task_a_evaluation(data_pred, data_true, print_out=False):
    """

    :param data_pred: Predicted data
    :type data_pred: list[list[dict[str, float|str]]]
    :param data_true: Ground truth data
    :type data_true: list[list[dict[str, float|str]]]
    :param print_out: Verbosity
    :type print_out: bool
    :return: Dict with `recall`, `precision`, and `f1`
    :rtype: dict[str, float]
    """
    ground_truth = FileFormat(data_true)
    prediction = FileFormat(data_pred)
    if print_out:
        print(ground_truth.compute_metrics_string(prediction))
    return ground_truth.compute_metrics(prediction)


def main():
    # Test code
    pass


if __name__ == '__main__':
    main()
