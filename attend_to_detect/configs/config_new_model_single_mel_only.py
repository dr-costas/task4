from torch.optim import *
from torch.nn import functional
from torch.nn.init import orthogonal, xavier_uniform, xavier_normal


class_freqs_alarm = [383., 341., 192., 260., 574., 2557., 2491., 1533., 695.]

class_freqs_vehicle = [2073., 1646., 27218., 3882., 3962., 7492., 3426., 2256.]

all_freqs_alarms_first = class_freqs_alarm + class_freqs_vehicle
all_freqs_vehicles_first = class_freqs_vehicle + class_freqs_alarm

# General variables
batch_size = 32
epochs = 300
dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 1.
network_loss_weight = False
lr_iterations = 900
lr_factor = .1

# Optimizer parameters
optimizer = Adadelta
# optimizer = Adam
optimizer_lr = 1e-6
l1_factor = 0.
l2_factor = 0.

network_channels_out = [128, 128, 128, 128]
network_cnn_kernel_sizes = [(5, 5), (5, 5), (5, 5), (5, 5)]
network_cnn_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
network_cnn_paddings = [(0, 0), (1, 1), (1, 1), (1, 1)]
network_cnn_activations = [functional.leaky_relu]

network_pool_kernels = [(2, 2), (3, 3), (3, 3), (3, 3)]
network_pool_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
network_pool_paddings = [(0, 0), (1, 1), (1, 1), (1, 1)]
#
# network_channels_out = [128, 128, 128]  # , 128]
# network_cnn_kernel_sizes = [(3, 3), (3, 3), (3, 3)]  # , (3, 3)]
# network_cnn_strides = [(2, 2), (2, 2), (2, 2)]  # , (2, 2)]
# network_cnn_paddings = [(1, 1), (1, 1), (1, 1)]  # , (1, 1)]
# network_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]  # , functional.leaky_relu]
#
# network_pool_kernels = [(3, 3), (3, 3), (3, 3)]  # , (3, 3)]
# network_pool_strides = [(2, 2), (2, 2), (2, 2)]  # , (2, 2)]
# network_pool_paddings = [(1, 1), (1, 1), (1, 1)]  # , (1, 1)]

network_rnn_input_size = 128
network_rnn_output_dims = [128, 128]  #, 256]

network_rnn_activations = [functional.tanh]

rnn_time_steps_out = 13

network_init = xavier_normal

network_dropout_cnn = 0.
network_dropout_rnn_input = 0.5
network_dropout_rnn_recurrent = 0.25

network_rnn_subsamplings = [1]

mlp_dims = [128, 64, len(all_freqs_vehicles_first)]
mlp_activations = [functional.relu, functional.relu, functional.tanh]
mlp_dropouts = [0.]

last_rnn_dim = len(all_freqs_vehicles_first)
last_rnn_activation = functional.tanh
last_rnn_dropout_i = 0.5
last_rnn_dropout_h = 0.

use_scaler = True

# EOF
