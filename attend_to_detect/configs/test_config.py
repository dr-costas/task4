from torch.optim import Adam
from torch.nn import functional
from torch.nn.init import orthogonal

# General variables
batch_size = 8
epochs = 300
dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 0.

# Optimizer parameters
optimizer = Adam
optimizer_lr = 1e-4

# Variables needed for the common feature extraction layer
common_out_channels = 40
common_kernel_size = (3, 3)
common_stride = (1, 1)
common_padding = (1, 1)
common_dropout = 0.5
common_dilation = (1, 1)
common_activation = functional.leaky_relu
common_max_pool_kernel = None
common_max_pool_stride = None
common_max_pool_padding = None


# Variables needed for the alarm branch
branch_alarm_channels_out = [40, 40, 40]
branch_alarm_cnn_kernel_sizes = [(1, 3), (1, 3), (1, 3)]
branch_alarm_cnn_strides = [(1, 2), (1, 2), (1, 2)]
branch_alarm_cnn_paddings = [(0, 0), (0, 0), (0, 0)]
branch_alarm_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

branch_alarm_pool_kernels = [(3, 2), (3, 2), (3, 2)]
branch_alarm_pool_strides = [(3, 2), (3, 2), (3, 2)]
branch_alarm_pool_paddings = [(0, 0), (0, 0), (0, 0)]

branch_alarm_rnn_input_size = 80
branch_alarm_rnn_output_dims = [64, 64]
branch_alarm_rnn_activations = [functional.tanh, functional.tanh]
branch_alarm_attention_bias = True
branch_alarm_init = orthogonal

branch_alarm_dropout_cnn = 0.2
branch_alarm_dropout_rnn_input = 0.2
branch_alarm_dropout_rnn_recurrent = 0.2

branch_alarm_rnn_subsamplings = [3]

branch_alarm_decoder_dim = 32

# Variables needed for the vehicle branch
branch_vehicle_channels_out = [40, 40, 40]
branch_vehicle_cnn_kernel_sizes = [(1, 3), (1, 3), (1, 3)]
branch_vehicle_cnn_strides = [(1, 2), (1, 2), (1, 2)]
branch_vehicle_cnn_paddings = [(0, 0), (0, 0), (0, 0)]
branch_vehicle_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

branch_vehicle_pool_kernels = [(3, 2), (3, 2), (3, 2)]
branch_vehicle_pool_strides = [(3, 2), (3, 2), (3, 2)]
branch_vehicle_pool_paddings = [(0, 0), (0, 0), (0, 0)]

branch_vehicle_rnn_input_size = 80
branch_vehicle_rnn_output_dims = [64, 64]
branch_vehicle_rnn_activations = [functional.tanh, functional.tanh]
branch_vehicle_attention_bias = True
branch_vehicle_init = orthogonal

branch_vehicle_dropout_cnn = 0.2
branch_vehicle_dropout_rnn_input = 0.2
branch_vehicle_dropout_rnn_recurrent = 0.2

branch_vehicle_rnn_subsamplings = [3]

branch_vehicle_decoder_dim = 32

