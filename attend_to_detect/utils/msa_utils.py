#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import numpy as np
import itertools
import numba

__docformat__ = 'reStructuredText'


def get_s(t_steps):
    """

    :param t_steps:
    :type t_steps: int
    :return: shape (X, t_steps)
    :rtype: numpy.core.multiarray.ndarray
    """
    l = [np.zeros(t_steps) - 1]
    for t_1 in range(t_steps):
        for t_2 in range(t_1 + 1, t_steps + 1):
            x = np.zeros(t_steps) - 1
            x[t_1:t_2] += 2
            l.append(x)
    return np.array(l)


def get_s_2(t_steps):
    """

    :param t_steps:
    :type t_steps: int
    :return: shape (X, t_steps)
    :rtype: numpy.core.multiarray.ndarray
    """
    l = []
    for i in itertools.product([-1, 1], repeat=t_steps):
        l.append(i)
    return np.array(l)


@numba.jit(nopython=True, nogil=True)
def find_max_mean(batch_array, target_classes, s):
    """

    :param batch_array: (b_size, t_steps, num_classes)
    :type batch_array: numpy.core.multiarray.ndarray
    :param s: shape (X, t_steps)
    :type s: numpy.core.multiarray.ndarray
    :param target_classes: (b_size, num_classes) | binary indication of classes
    :type target_classes: numpy.core.multiarray.ndarray
    :return:
    :rtype:
    """
    b_size = batch_array.shape[0]

    num_classes = target_classes.shape[-1]

    max_means = np.zeros((b_size, num_classes))
    max_means_index_start = np.zeros((b_size, num_classes))
    max_means_index_end = np.zeros((b_size, num_classes))
    s_i_indices = np.zeros((b_size, num_classes))

    for b_index in range(b_size):

        for class_index in range(num_classes):

            class_array = np.copy(batch_array[b_index, :, class_index])

            if target_classes[b_index, class_index] == 0:
                class_array *= -1
                max_means[b_index, class_index] = class_array.mean()
                max_means_index_start[b_index, class_index] = 0
                max_means_index_end[b_index, class_index] = 0
                s_i_indices[b_index, class_index] = -1

            else:
                tmp_max_means = np.zeros(len(s))
                tmp_max_means_start = np.zeros(len(s))
                tmp_max_means_end = np.zeros(len(s))
                s_i_indices_tmp = np.zeros(len(s))

                for s_i in range(len(s)):
                    local_array = class_array * s[s_i, :]
                    tmp_max_means[s_i] = local_array.mean()
                    g = s[s_i, :]
                    ind = np.where(g == 1)
                    indices = ind[0]
                    if len(indices) == 0:
                        tmp_max_means_start[s_i] = 0
                        tmp_max_means_end[s_i] = 0
                    else:
                        tmp_max_means_start[s_i] = indices[0]
                        tmp_max_means_end[s_i] = indices[-1]
                    s_i_indices_tmp[s_i] = s_i

                max_index = np.argmax(tmp_max_means)
                max_means[b_index, class_index] = tmp_max_means[max_index]
                max_means_index_start[b_index, class_index] = tmp_max_means_start[max_index]
                max_means_index_end[b_index, class_index] = tmp_max_means_end[max_index]
                s_i_indices[b_index, class_index] = s_i_indices_tmp[max_index]

    return max_means, s_i_indices, max_means_index_end, max_means_index_end


def find_max_mean_2(batch_array, target_classes, s):
    """

    :param batch_array: (b_size, t_steps, num_classes)
    :type batch_array: numpy.core.multiarray.ndarray
    :param s: shape (X, t_steps)
    :type s: numpy.core.multiarray.ndarray
    :param target_classes: (b_size, num_classes) | binary indication of classes
    :type target_classes: numpy.core.multiarray.ndarray
    :return:
    :rtype:
    """
    b_size = batch_array.shape[0]

    num_classes = target_classes.shape[-1]

    max_means = np.zeros((b_size, num_classes))
    max_means_index_start = np.zeros((b_size, num_classes))
    max_means_index_end = np.zeros((b_size, num_classes))

    for b_index in range(b_size):

        for class_index in range(num_classes):

            class_array = np.copy(batch_array[b_index, :, class_index])

            if target_classes[b_index, class_index] == 0:
                class_array *= -1
                max_means[b_index, class_index] = class_array.mean()
                max_means_index_start[b_index, class_index] = 0
                max_means_index_end[b_index, class_index] = 0

            else:
                tmp_max_means = np.zeros(len(s))
                tmp_max_means_start = np.zeros(len(s))
                tmp_max_means_end = np.zeros(len(s))

                for s_i in range(len(s)):
                    local_array = class_array * s[s_i, :]
                    tmp_max_means[s_i] = local_array.mean()
                    g = s[s_i, :]
                    ind = np.where(g == 1)
                    indices = ind[0]
                    if len(indices) == 0:
                        tmp_max_means_start[s_i] = 0
                        tmp_max_means_end[s_i] = 0
                    else:
                        tmp_max_means_start[s_i] = indices[0]
                        tmp_max_means_end[s_i] = indices[-1]

                max_index = np.argmax(tmp_max_means)
                max_means[b_index, class_index] = tmp_max_means[max_index]
                max_means_index_start[b_index, class_index] = tmp_max_means_start[max_index]
                max_means_index_end[b_index, class_index] = tmp_max_means_end[max_index]

    return max_means, max_means_index_end, max_means_index_end


def main():
    import time

    t_steps = 10
    num_classes = 17
    b_size = 64

    print('-' * 50)
    print('Checking function `get_s` for t_steps = {}'.format(t_steps))
    s_time = time.time()
    s = get_s_2(t_steps)
    print('Elapsed time: {}\n'.format(time.time() - s_time))
    print('Shape of returned array: {}'.format(s.shape))
    rand_ints = np.random.randint(0, s.shape[0], 3)
    print('The output of the three first [0 1 2] and three randomly '
          'selected lines {}'.format(rand_ints))
    for i in range(3):
        print(s[i, :])
    for i in rand_ints:
        print(s[i, :])

    print(' ')
    print('-' * 50)
    b_array = (np.random.rand(b_size, t_steps, num_classes) * 2) - 1
    print('Checking function `find_max_mean` for '
          '(batch_size, t_steps, num_classes) = {}'.format(b_array.shape))
    y_true = np.random.randint(0, 2, (b_size, num_classes))
    # means_array, starting_indices, ending_indices = find_max_mean(b_array, y_true, s)
    s_time = time.time()
    means_array, starting_indices, ending_indices = find_max_mean(b_array, y_true, s)
    print('Elapsed time: {}'.format(time.time() - s_time))
    print('Shape of returned array: {}'.format(means_array.shape))
    means_array_2, starting_indices_2, ending_indices_2 = find_max_mean_2(b_array, y_true, s)
    print(np.all(np.isclose(means_array, means_array_2)))
    print(np.array_equal(starting_indices, starting_indices_2))
    print(np.array_equal(ending_indices, ending_indices_2))


if __name__ == '__main__':
    main()

# EOF
