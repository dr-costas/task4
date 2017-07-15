from torch.optim import Adam
from torch.nn import functional
from torch.nn.init import orthogonal

# General variables
batch_size = 16
epochs = 300
dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 0.
alarm_loss_weight = 0.01
vehicle_loss_weight = None  # None for no weight

# Optimizer parameters
optimizer = Adam
optimizer_lr = 1e-4

# Variables needed for the common feature extraction layer
common_out_channels = 32
common_kernel_size = (5, 5)
common_stride = (1, 1)
common_padding = (0, 0)
common_dropout = 0.5
common_dilation = (1, 1)
common_activation = functional.leaky_relu
common_max_pool_kernel = (5, 5)
common_max_pool_stride = (1, 2)
common_max_pool_padding = (0, 1)


# Variables needed for the alarm branch
branch_alarm_channels_out = [64, 128, 256]
branch_alarm_cnn_kernel_sizes = [(3, 5), (3, 3), (3, 3)]
branch_alarm_cnn_strides = [(1, 2), (2, 2), (2, 2)]
branch_alarm_cnn_paddings = [(0, 1), (0, 1), (0, 0)]
branch_alarm_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

branch_alarm_pool_kernels = [(5, 5), (5, 5), (5, 5)]
branch_alarm_pool_strides = [(2, 2), (2, 2), (2, 3)]
branch_alarm_pool_paddings = [(1, 1), (0, 1), (0, 2)]

branch_alarm_rnn_input_size = 256
branch_alarm_rnn_output_dims = [128, 128]
branch_alarm_rnn_activations = [functional.tanh, functional.tanh]
branch_alarm_attention_bias = True
branch_alarm_init = orthogonal

branch_alarm_dropout_cnn = 0.25
branch_alarm_dropout_rnn_input = 0.25
branch_alarm_dropout_rnn_recurrent = 0.0

branch_alarm_rnn_subsamplings = [1]

branch_alarm_decoder_dim = 128

# Variables needed for the vehicle branch
branch_vehicle_channels_out = [64, 128, 256]
branch_vehicle_cnn_kernel_sizes = [(3, 5), (3, 3), (3, 3)]
branch_vehicle_cnn_strides = [(1, 2), (2, 2), (2, 2)]
branch_vehicle_cnn_paddings = [(0, 1), (0, 1), (0, 0)]
branch_vehicle_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

branch_vehicle_pool_kernels = [(5, 5), (5, 5), (5, 5)]
branch_vehicle_pool_strides = [(2, 2), (2, 2), (2, 3)]
branch_vehicle_pool_paddings = [(1, 1), (0, 1), (0, 2)]

branch_vehicle_rnn_input_size = 256
branch_vehicle_rnn_output_dims = [128, 128]
branch_vehicle_rnn_activations = [functional.tanh, functional.tanh]
branch_vehicle_attention_bias = True
branch_vehicle_init = orthogonal

branch_vehicle_dropout_cnn = 0.5
branch_vehicle_dropout_rnn_input = 0.5
branch_vehicle_dropout_rnn_recurrent = 0.25

branch_vehicle_rnn_subsamplings = [1]

branch_vehicle_decoder_dim = 128

