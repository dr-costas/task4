from torch.optim import *
from torch.nn import functional
from torch.nn.init import orthogonal, xavier_uniform, xavier_normal

# General variables
batch_size = 32
epochs = 300
dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 1.
network_loss_weight = True
lr_iterations = 900
lr_factor = .1

# Optimizer parameters
optimizer = Adam
# optimizer = Adam
optimizer_lr = 1e-4
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
network_decoder_dim = 256

network_rnn_activations = [functional.tanh]
network_attention_bias = True
network_init = xavier_normal

network_dropout_cnn = 0.
network_dropout_rnn_input = 0.5
network_dropout_rnn_recurrent = 0.25

network_rnn_subsamplings = [1]

use_scaler = True

# EOF
