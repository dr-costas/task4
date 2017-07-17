from torch.optim import Adam
from torch.nn import functional
from torch.nn.init import orthogonal, xavier_uniform

# General variables
batch_size = 32
epochs = 300
dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 0.
network_loss_weight = None

# Optimizer parameters
optimizer = Adam
optimizer_lr = 1e-3

# Variables needed for the alarm branch
network_channels_out = [64, 64, 128, 128]
network_cnn_kernel_sizes = [(5, 5), (3, 5), (3, 3), (3, 3)]
network_cnn_strides = [(1, 1), (1, 2), (2, 2), (2, 2)]
network_cnn_paddings = [(0, 0), (0, 1), (0, 1), (0, 0)]
network_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

network_pool_kernels = [(5, 5), (5, 5), (5, 5), (5, 5)]
network_pool_strides = [(1, 2), (2, 2), (2, 2), (2, 3)]
network_pool_paddings = [(0, 1), (1, 1), (0, 1), (0, 2)]

network_rnn_input_size = 128
network_rnn_output_dims = [128, 128]
network_rnn_activations = [functional.tanh, functional.tanh]
network_attention_bias = True
network_init = xavier_uniform

network_dropout_cnn = 0.25
network_dropout_rnn_input = 0.25
network_dropout_rnn_recurrent = 0.0

network_rnn_subsamplings = [1]

network_decoder_dim = 128

# EOF
