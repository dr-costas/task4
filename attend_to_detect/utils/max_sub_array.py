import numpy as np
import time


__docformat__ = 'reStructuredText'


def max_sub_array(the_array):
    """Kadane's algorithm

    :param the_array: 3d array of scores (batch_size, n_steps, n_classes)
    :type the_array: numpy.core.multiarray.ndarray
    :return:
    :rtype:
    """
    batch_size, n_steps, n_classes = the_array.shape
    max_ending_here = the_array[:, 0, :].copy()
    max_so_far = the_array[:, 0, :].copy()

    max_end_index = np.zeros((batch_size, n_classes))
    max_start_index = np.zeros((batch_size, n_classes))
    max_start_index_potential = np.zeros((batch_size, n_classes))

    for i in range(1, n_steps):
        x = the_array[:, i, :]

        max_ending_here = np.maximum(x, max_ending_here + x)
        max_start_index_potential[max_ending_here == x] = i

        max_so_far = np.maximum(max_so_far, max_ending_here)

        tmp_indices = max_so_far == max_ending_here
        max_end_index[tmp_indices] = i
        max_start_index[tmp_indices] = max_start_index_potential[tmp_indices]

    return max_so_far, max_start_index, max_end_index


def max_sub_array_single(the_array):
    """

    :param the_array:
    :type the_array: numpy.core.multiarray.ndarray
    :return:
    :rtype:
    """
    max_ending_here = max_so_far = the_array[0]

    max_start_idx_potential = 0
    max_end_idx_cur = 0
    max_start_idx_cur = 0

    for i in range(1, the_array.shape[0]):
        x = the_array[i]
        max_ending_here = np.maximum(x, max_ending_here + x)
        if max_ending_here == x:
            max_start_idx_potential = i

        max_so_far = np.maximum(max_so_far, max_ending_here)
        if max_so_far == max_ending_here:
            max_end_idx_cur = i
            max_start_idx_cur = max_start_idx_potential

    return max_so_far, (max_start_idx_cur, max_end_idx_cur)


def main():
    x = np.array([-1, -1, 10, 10, -100, -1, 9, 9, -1])
    print(max_sub_array_single(x))

    x = np.array(
        [
            [
                [-1, -1, 10, 10, -100, -1, 9, 9, -1]
            ],
            [
                [-1, 10, 10, -100, -1, 9, 9, 9, -1]
            ]
        ]
    )

    x = x.transpose((1, 2, 0))
    print(x.shape)
    print(x[0])

    r, s, e = max_sub_array(x)

    print(s)
    print(e)

    x = np.random.rand(64, 27, 17)

    s_time = time.time()
    r, s, e = max_sub_array(x)
    print(time.time() - s_time)

if __name__ == '__main__':
    main()
