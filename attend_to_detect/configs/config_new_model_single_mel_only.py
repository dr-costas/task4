from torch.optim import *
from torch.nn import functional
from torch.nn.init import orthogonal, xavier_uniform, xavier_normal


class_freqs_alarm = [383., 341., 192., 260., 574., 2557., 2491., 1533., 695.]

class_freqs_vehicle = [2073., 1646., 27218., 3882., 3962., 7492., 3426., 2256.]

all_freqs_alarms_first = class_freqs_alarm + class_freqs_vehicle
all_freqs_vehicles_first = class_freqs_vehicle + class_freqs_alarm

# General variables
batch_size = 64
epochs = 300
lr_iterations = 800

dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'
dataset_local_path = '/Tmp/drososko/dcase_2017_task_4_test.hdf5'

use_scaler = True
network_loss_weight = True
# weighting_factor = class_freqs_vehicle[2]
weighting_factor = 64 * 800
find_max_mean_formulation = 4

# Optimizer parameters
grad_clip_norm = 0.
# optimizer = Adadelta
# optimizer_dict = {
#     'lr': 1e-5
# }
# optimizer = Adam
# optimizer_dict = {
#     'lr': 1e-4
# }
optimizer = SGD
optimizer_dict = {
    'lr': 1e-5,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 1e-4
}

lr_factor = .95
l1_factor = 0.
l2_factor = 0.

nb_features = 64

# MEL only 1
# network_channels_out = [64, 64, 64, 64]
# network_cnn_kernel_sizes = [(5, 5), (5, 5), (5, 5), (5, 5)]
# network_cnn_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
# network_cnn_paddings = [(0, 0), (1, 1), (1, 1), (1, 1)]
# network_cnn_activations = [functional.leaky_relu]
#
# network_pool_kernels = [(2, 2), (3, 3), (3, 3), (3, 3)]
# network_pool_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
# network_pool_paddings = [(0, 0), (1, 1), (1, 1), (1, 1)]
# rnn_time_steps_out = 13
#

# MEL only 2
network_channels_out = [256, 256, 256, 256, 256]
network_cnn_kernel_sizes = [(3, 3)] * len(network_channels_out)
network_cnn_strides = [(2, 2)] * len(network_channels_out)
network_cnn_paddings = [(1, 1)] * len(network_channels_out)
network_cnn_activations = [functional.leaky_relu]

network_pool_kernels = [(1, 1), (1, 1), (1, 1), (3, 3), (1, 1)]
network_pool_strides = [(1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
network_pool_paddings = [(0, 0), (0, 0), (0, 0), (1, 1), (0, 0)]
rnn_time_steps_out = 14
#

# MEL and Delta_1
# network_channels_out = [32, 64, 128]
# network_cnn_kernel_sizes = [(3, 3)] * 3
# network_cnn_strides = [(2, 2)] * 3
# network_cnn_paddings = [(1, 1)] * 3
# network_cnn_activations = [functional.leaky_relu]
#
# network_pool_kernels = [(3, 3)] * 3
# network_pool_strides = [(2, 2)] * 3
# network_pool_paddings = [(1, 1)] * 3
# rnn_time_steps_out = 14

network_rnn_input_size = 128
network_rnn_output_dims = [256, 256]  #, 256]

network_rnn_activations = [functional.tanh]


network_init = xavier_uniform

network_dropout_cnn = 0.5
network_dropout_rnn_input = 0.5
network_dropout_rnn_recurrent = 0.5

network_rnn_subsamplings = [1]

# network_rnn_input_size = 256
# network_rnn_output_dims = [128, 64]  #, 256]
#
# network_rnn_activations = [functional.tanh]
#
#
# network_init = xavier_uniform
#
# network_dropout_cnn = 0.5
# network_dropout_rnn_input = 0.5
# network_dropout_rnn_recurrent = 0.5
#
# network_rnn_subsamplings = [1]

mlp_dims = [256, 128, len(all_freqs_vehicles_first)]
mlp_activations = [functional.tanh, functional.tanh, functional.tanh]
mlp_dropouts = [0.5]

last_rnn_dim = len(all_freqs_vehicles_first)
last_rnn_activation = functional.tanh
last_rnn_dropout_i = 0.5
last_rnn_dropout_h = 0.5
last_rnn_extra_activation = None

# EOF
