# -*- coding: utf-8 -*-

# imports
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant, orthogonal
import numpy as np

__docformat__ = 'reStructuredText'


def make_me_a_list(with_that, and_this_long=1):
    if type(with_that) is not list:
        with_that = [with_that]

    if len(with_that) == 1:
        with_that = with_that * and_this_long

    return with_that


class CategoryBranch2(nn.Module):

    def __init__(self,
                 cnn_channels_in, cnn_channels_out,
                 cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_activations,
                 max_pool_kernels, max_pool_strides, max_pool_paddings,
                 rnn_input_size, rnn_out_dims, rnn_activations,
                 dropout_cnn, dropout_rnn_input, dropout_rnn_recurrent,
                 rnn_subsamplings, rnn_t_steps_out,
                 mlp_dims, mlp_activations, mlp_dropouts,
                 last_rnn_dim, last_rnn_activation, last_rnn_dropout_i,
                 last_rnn_dropout_h, init=xavier_normal):

        super(CategoryBranch2, self).__init__()

        if type(cnn_channels_out) is not list:
            raise AttributeError('Channels out must be a list of len '
                                 'equal to the desired amount of conv '
                                 'layers')

        self.cnn_channels_out = [cnn_channels_in] + cnn_channels_out

        cnn_len = len(self.cnn_channels_out) - 1

        self.cnn_kernel_sizes = make_me_a_list(cnn_kernel_sizes, cnn_len)
        self.cnn_strides = make_me_a_list(cnn_strides, cnn_len)
        self.cnn_paddings = make_me_a_list(cnn_paddings, cnn_len)
        self.cnn_activations = make_me_a_list(cnn_activations, cnn_len)

        self.max_pool_kernels = make_me_a_list(max_pool_kernels, cnn_len)
        self.max_pool_strides = make_me_a_list(max_pool_strides, cnn_len)
        self.max_pool_paddings = make_me_a_list(max_pool_paddings, cnn_len)

        self.rnn_out_dims = [rnn_input_size] + rnn_out_dims
        rnn_len = len(self.rnn_out_dims) - 1
        self.rnn_activations_f = make_me_a_list(rnn_activations, rnn_len)
        self.rnn_activations_b = make_me_a_list(rnn_activations, rnn_len)

        self.rnn_subsamplings = make_me_a_list(rnn_subsamplings, rnn_len - 1) + [1]

        self.dropout_cnn = dropout_cnn
        self.dropout_rnn_input_f = dropout_rnn_input
        self.dropout_rnn_input_b = dropout_rnn_input
        self.dropout_rnn_recurrent_f = dropout_rnn_recurrent
        self.dropout_rnn_recurrent_b = dropout_rnn_recurrent

        if not type(mlp_dims) in [list, tuple]:
            raise AttributeError('MLP dimensions should be a list')

        mlp_input_dim = self.rnn_out_dims[-1] * 2 * rnn_t_steps_out

        self.mlp_dims = [mlp_input_dim] + mlp_dims
        self.mlp_activations = make_me_a_list(mlp_activations, len(self.mlp_dims) - 1)
        self.mlp_dropouts_values = make_me_a_list(mlp_dropouts, len(self.mlp_dims) - 1)

        all_cnn_have_proper_len = all([
            len(a) == cnn_len
            for a in [
                self.cnn_kernel_sizes, self.cnn_strides,
                self.cnn_paddings, self.cnn_activations
            ]
        ])

        if not all_cnn_have_proper_len:
            raise AttributeError('Either provide arguments for all layers or '
                                 'just one (except of channels out)')

        if not len(self.rnn_activations_f) == len(self.rnn_out_dims) - 1:
            raise AttributeError('Amount of rnn activations should be equal to '
                                 'the specified output dims of rnns')

        self.cnn_layers = []
        self.cnn_dropout_layers = []
        self.bn_layers = []
        self.pooling_layers = []
        self.rnn_layers_f = []
        self.rnn_layers_b = []
        self.rnn_dropout_layers_input_f = []
        self.rnn_dropout_layers_recurrent_f = []
        self.rnn_dropout_layers_input_b = []
        self.rnn_dropout_layers_recurrent_b = []
        self.mlps = []
        self.mlps_dropouts = []

        for i in range(len(self.cnn_channels_out) - 1):
            self.cnn_layers.append(nn.Conv2d(
                in_channels=self.cnn_channels_out[i],
                out_channels=self.cnn_channels_out[i + 1],
                kernel_size=cnn_kernel_sizes[i],
                stride=self.cnn_strides[i],
                padding=self.cnn_paddings[i]
            ))
            self.bn_layers.append(nn.BatchNorm2d(
                num_features=self.cnn_channels_out[i + 1]
            ))
            self.pooling_layers.append(nn.MaxPool2d(
                kernel_size=self.max_pool_kernels[i],
                stride=self.max_pool_strides[i],
                padding=self.max_pool_paddings[i]
            ))
            self.cnn_dropout_layers.append(
                nn.Dropout2d(self.dropout_cnn)
            )

            setattr(self, 'cnn_layer_{}'.format(i+1), self.cnn_layers[-1])
            setattr(self, 'bn_layer_{}'.format(i+1), self.bn_layers[-1])
            setattr(self, 'pool_layer_{}'.format(i+1), self.pooling_layers[-1])
            setattr(self, 'cnn_dropout_layer_{}'.format(i+1), self.cnn_dropout_layers[-1])
            setattr(self, 'cnn_activation_{}'.format(i + 1), self.cnn_activations[i])

        for i in range(len(self.rnn_out_dims) - 1):
            if i > 0:
                # output of last layer was bidirectional, input is bigger
                input_size = 2 * self.rnn_out_dims[i]
            else:
                input_size = self.rnn_out_dims[i]
            self.rnn_layers_f.append(nn.GRUCell(
                input_size=input_size,
                hidden_size=self.rnn_out_dims[i + 1]
            ))
            self.rnn_layers_b.append(nn.GRUCell(
                input_size=input_size,
                hidden_size=self.rnn_out_dims[i + 1]
            ))
            self.rnn_dropout_layers_input_f.append(
                nn.Dropout(self.dropout_rnn_input_f))
            self.rnn_dropout_layers_recurrent_f.append(
                nn.Dropout(self.dropout_rnn_recurrent_f))

            self.rnn_dropout_layers_input_b.append(
                nn.Dropout(self.dropout_rnn_input_b))
            self.rnn_dropout_layers_recurrent_b.append(
                nn.Dropout(self.dropout_rnn_recurrent_b))

            setattr(self, 'rnn_layer_f_{}'.format(i+1), self.rnn_layers_f[-1])
            setattr(self, 'rnn_layer_b_{}'.format(i+1), self.rnn_layers_b[-1])

            setattr(self, 'rnn_dropout_input_f_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_input_f[-1])
            setattr(self, 'rnn_dropout_recurrent_f_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_recurrent_f[-1])
            setattr(self, 'rnn_activation_f_{}'.format(i + 1),
                    self.rnn_activations_f[i])

            setattr(self, 'rnn_dropout_input_b_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_input_b[-1])
            setattr(self, 'rnn_dropout_recurrent_b_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_recurrent_b[-1])
            setattr(self, 'rnn_activation_b_{}'.format(i + 1),
                    self.rnn_activations_b[i])

        for i in range(len(self.mlp_dims) - 1):
            self.mlps.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i+1]))
            self.mlps_dropouts.append(nn.Dropout(self.mlp_dropouts_values[i]))
            setattr(self, 'mlp_{}'.format(i + 1), self.mlps[-1])
            setattr(self, 'mlp_dropout_{}'.format(i + 1), self.mlps_dropouts[-1])
            setattr(self, 'mlp_activation_{}'.format(i + 1), self.mlp_activations[i])

        self.last_rnn_dim = last_rnn_dim
        self.last_rnn_layer = nn.GRUCell(self.rnn_out_dims[-1] * 2, self.last_rnn_dim)
        self.last_rnn_dropout_i = nn.Dropout(last_rnn_dropout_i)
        self.last_rnn_dropout_h = nn.Dropout(last_rnn_dropout_h)
        self.last_rnn_activation = last_rnn_activation

        self.initialize(init)

    @staticmethod
    def init_gru_cell(for_module, init):
        init(for_module.weight_ih.data)
        init(for_module.weight_hh.data)
        constant(for_module.bias_ih.data, 0)
        constant(for_module.bias_hh.data, 0)

    def initialize(self, init):
        # Initialize CNNs
        for a_cnn_layer in self.cnn_layers:
            xavier_normal(a_cnn_layer.weight.data)
            constant(a_cnn_layer.bias.data, 0)

        # Initialize RNNs
        for an_rnn_layer in self.rnn_layers_f + self.rnn_layers_b + [self.last_rnn_layer]:
            self.init_gru_cell(an_rnn_layer, init)

        # Initialize MLPs
        for a_linear_layer in self.mlps:
            xavier_normal(a_linear_layer.weight.data)
            constant(a_linear_layer.bias.data, 0)

    def get_initial_decoder_state(self, batch_size):
        # TODO: smarter initial state
        state = Variable(torch.zeros((batch_size, self.decoder_dim)))
        if torch.has_cudnn:
            state = state.cuda()
        return state

    def forward(self, x):

        output = self.pooling_layers[0](self.bn_layers[0](
            self.cnn_activations[0](self.cnn_layers[0](
                self.cnn_dropout_layers[0](x)
            ))
        ))

        for dropout, pooling, bn, activation, cnn in zip(
                self.cnn_dropout_layers[1:],
                self.pooling_layers[1:],
                self.bn_layers[1:],
                self.cnn_activations[1:],
                self.cnn_layers[1:]):
            output = pooling(bn(activation(cnn(dropout(output)))))

        output = output.permute(0, 2, 1, 3)
        o_size = output.size()
        output = output.resize(o_size[0], o_size[1], o_size[2] * o_size[3])
        o_size = output.size()

        for i in range(len(self.rnn_layers_f)):
            h_s_f = Variable(torch.zeros(o_size[0], o_size[1], self.rnn_out_dims[i+1]))
            h_s_b = Variable(torch.zeros(o_size[0], o_size[1], self.rnn_out_dims[i+1]))

            zeros_f = Variable(torch.zeros(o_size[0], self.rnn_out_dims[i+1]))
            zeros_b = Variable(torch.zeros(o_size[0], self.rnn_out_dims[i+1]))

            if torch.has_cudnn:
                h_s_f = h_s_f.cuda()
                h_s_b = h_s_b.cuda()
                zeros_f = zeros_f.cuda()
                zeros_b = zeros_b.cuda()

            h_s_f[:, 0, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                self.rnn_dropout_layers_input_f[i](output[:, 0, :]),
                zeros_f))
            h_s_b[:, -1, :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                self.rnn_dropout_layers_input_b[i](output[:, -1, :]),
                zeros_b))

            for s_i in range(1, o_size[1]):
                h_s_f[:, s_i, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                    self.rnn_dropout_layers_input_f[i](output[:, s_i, :]),
                    self.rnn_dropout_layers_recurrent_f[i](h_s_f[:, s_i - 1, :])
                ))
                h_s_b[:, -(s_i + 1), :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                    self.rnn_dropout_layers_input_b[i](output[:, -(s_i + 1), :]),
                    self.rnn_dropout_layers_recurrent_b[i](h_s_b[:, s_i - 1, :])
                ))

            output = torch.cat([h_s_f, h_s_b], -1)
            o_size = output.size()
            u_l = o_size[1]
            u_l -= divmod(o_size[1], self.rnn_subsamplings[i])[-1]
            output = output[:, 0:u_l:self.rnn_subsamplings[i], :]
            o_size = output.size()

        mlp_output = self.mlp_activations[0](
            self.mlps[0](self.mlps_dropouts[0](
                output.view(o_size[0], o_size[1] * o_size[2]))))

        if len(self.mlps) > 1:
            for mlp, activation, dropout in zip(
                    self.mlps[1:], self.mlp_activations[1:], self.mlps_dropouts[1:]):
                mlp_output = activation(mlp(dropout(mlp_output)))

        h_s = Variable(torch.zeros(o_size[0], o_size[1], self.last_rnn_dim))
        zeros = Variable(torch.zeros(o_size[0], self.last_rnn_dim))

        if torch.has_cudnn:
            h_s = h_s.cuda()
            zeros = zeros.cuda()

        h_s[:, 0, :] = nn.functional.sigmoid(self.last_rnn_activation(self.last_rnn_layer(
            self.last_rnn_dropout_i(output[:, 0, :]),
            zeros
        )))

        for s_i in range(1, o_size[1]):
            h_s[:, s_i, :] = nn.functional.sigmoid(self.last_rnn_activation(self.last_rnn_layer(
                self.last_rnn_dropout_i(output[:, s_i, :]),
                self.last_rnn_dropout_h(h_s[:, s_i - 1, :])
            )))

        return h_s, mlp_output

    def nb_trainable_parameters(self):
        nb_params = 0
        for param in self.parameters():
            nb_params += np.product(param.size())

        return nb_params


def main():
    from torch.autograd import Variable
    from torch.nn import functional
    x = Variable(
        torch.rand(2, 1, 862, 64).float()
    )

    nb_cnn_layers = 5

    b = CategoryBranch2(
        cnn_channels_in=1,
        cnn_channels_out=[40] * nb_cnn_layers,
        cnn_kernel_sizes=[(5, 5)] * nb_cnn_layers,
        cnn_strides=[(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)],
        cnn_paddings=[(0, 0), (1, 1), (1, 1), (1, 1), (1, 1)],
        cnn_activations=[functional.leaky_relu],
        max_pool_kernels=[(1, 1), (1, 1), (3, 3), (1, 1), (3, 3)],
        max_pool_strides=[(1, 1), (1, 1), (2, 2), (1, 1), (2, 2)],
        max_pool_paddings=[(0, 0), (0, 0), (1, 1), (0, 0), (1, 1)],
        rnn_input_size=40,
        rnn_out_dims=[64] * 2,
        rnn_activations=[functional.tanh],
        dropout_cnn=0.2,
        dropout_rnn_input=0.2,
        dropout_rnn_recurrent=0.2,
        rnn_subsamplings=[1],
        rnn_t_steps_out=13,
        mlp_dims=[34, 17],
        mlp_activations=[nn.functional.tanh],
        mlp_dropouts=[0.5],
        last_rnn_dim=17,
        last_rnn_activation=nn.functional.tanh,
        last_rnn_dropout_i=0.5,
        last_rnn_dropout_h=0.2,
    )

    rnn_out, mlp_out = b(x)

    print(rnn_out.size())
    print(mlp_out.size())
    print(b.nb_trainable_parameters())


if __name__ == '__main__':
    main()

# EOF
