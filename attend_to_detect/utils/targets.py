import torch

def generate_targets(subsequences, positive_classes, n_timesteps, n_classes):
    '''Generates a target matrix for a list of subsequences.
    subsequences: list of lists of tuples, where each tuple contains the
    indices of the maximum subsequence for each class.
    classes: list of tuples containing indices for the positive classes
    (i.e., categorical targets for a given sample)
    Output:is a FloatTensor with Size(batch_size, num_timesteps, num_classes).
    '''
    y = torch.FloatTensor(len(subsequences), n_timesteps, n_classes).fill_(-1.0)
    for k in range(len(subsequences)):
        for class_idx in positive_classes[k]:
            try:
                i0, i1 = subsequences[k][class_idx]
                y[k, i0:i1, class_idx] = 1.0
            except ValueError:
                # We did not get a valid range, push all targets to zero
                y[k, :, class_idx] = 0.0
    return y

