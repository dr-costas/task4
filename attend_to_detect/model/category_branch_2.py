#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import torch
from torch.autograd import Variable

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


def make_me_a_list(with_that, and_this_long=1):
    if type(with_that) is not list:
        with_that = [with_that]

    if len(with_that) == 1:
        with_that = with_that * and_this_long

    return with_that


class CategoryBranch2(torch.nn.Module):

    def __init__(self,
                 cnn_channels_in, cnn_channels_out,
                 cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_activations,
                 max_pool_kernels, max_pool_strides, max_pool_paddings,
                 rnn_input_size, rnn_out_dims, rnn_activations,
                 dropout_cnn, dropout_rnn_input, dropout_rnn_recurrent):

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

        self.dropout_cnn = dropout_cnn
        self.dropout_rnn_input_f = dropout_rnn_input
        self.dropout_rnn_input_b = dropout_rnn_input
        self.dropout_rnn_recurrent_f = dropout_rnn_recurrent
        self.dropout_rnn_recurrent_b = dropout_rnn_recurrent

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

        for i in range(len(self.cnn_channels_out) - 1):
            self.cnn_layers.append(torch.nn.Conv2d(
                in_channels=self.cnn_channels_out[i],
                out_channels=self.cnn_channels_out[i + 1],
                kernel_size=cnn_kernel_sizes[i],
                stride=self.cnn_strides[i],
                padding=self.cnn_paddings[i]
            ))
            self.bn_layers.append(torch.nn.BatchNorm2d(
                num_features=self.cnn_channels_out[i + 1]
            ))
            self.pooling_layers.append(torch.nn.MaxPool2d(
                kernel_size=self.max_pool_kernels[i],
                stride=self.max_pool_strides[i],
                padding=self.max_pool_paddings[i]
            ))
            self.cnn_dropout_layers.append(
                torch.nn.Dropout2d(self.dropout_cnn)
            )

            setattr(self, 'cnn_layer_{}'.format(i+1), self.cnn_layers[-1])
            setattr(self, 'bn_layer_{}'.format(i+1), self.bn_layers[-1])
            setattr(self, 'pool_layer_{}'.format(i+1), self.pooling_layers[-1])
            setattr(self, 'cnn_dropout_layer_{}'.format(i+1), self.cnn_dropout_layers[-1])
            setattr(self, 'cnn_activation_{}'.format(i + 1), self.cnn_activations[i])

        for i in range(len(self.rnn_out_dims) - 1):
            self.rnn_layers_f.append(torch.nn.GRUCell(
                input_size=self.rnn_out_dims[i],
                hidden_size=self.rnn_out_dims[i + 1]
            ))
            self.rnn_layers_b.append(torch.nn.GRUCell(
                input_size=self.rnn_out_dims[i],
                hidden_size=self.rnn_out_dims[i + 1]
            ))
            self.rnn_dropout_layers_input_f.append(
                torch.nn.Dropout(self.dropout_rnn_input_f))
            self.rnn_dropout_layers_recurrent_f.append(
                torch.nn.Dropout(self.dropout_rnn_recurrent_f))

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

        for i in range(len(self.rnn_layers)):
            h_f = Variable(torch.zeros(o_size[0], o_size[1], self.rnn_out_dims[i]))
            h_b = Variable(torch.zeros(o_size[0], o_size[1], self.rnn_out_dims[i]))

            h_f[:, 0, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                self.rnn_dropout_layers_input_f[i](output[:, s_i, :]), h_f[:, 0, :]))
            h_b[:, 0, :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                self.rnn_dropout_layers_input_b[i](output[:, -1, :]), h_b[:, 0, :]))

            for s_i in range(1, o_size[1]):
                h_f[:, s_i, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                    self.rnn_dropout_layers_input_f[i](output[:, s_i, :]),
                    self.rnn_dropout_layers_recurrent_f[i](h_f[:, s_i - 1, :])
                ))
                h_b[:, s_i, :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                    self.rnn_dropout_layers_input_b[i](output[:, -(s_i + 1), :]),
                    self.rnn_dropout_layers_recurrent_b[i](h_f[:, s_i - 1, :])
                ))

            output = h_f + h_b
            o_size = output.size()
            u_l = o_size[1]
            u_l -= divmod(o_size[1], 2)[-1]
            output = output[:, 0:u_l:2, :]
            o_size = output.size()

        return output


def main():
    from torch.autograd import Variable
    from torch.nn import functional
    x = Variable(
        torch.rand(2, 1, 862, 192).float()
    )

    b = CategoryBranch2(
        cnn_channels_in=1,
        cnn_channels_out=[40] * 4,
        cnn_kernel_sizes=[(1, 3)] * 4,
        cnn_strides=[(1, 2)] * 4,
        cnn_paddings=[(0, 0)] * 4,
        cnn_activations=functional.leaky_relu,
        max_pool_kernels=[(3, 2)] * 4,
        max_pool_strides=[(3, 2)] * 4,
        max_pool_paddings=[(0, 0)] * 4,
        rnn_input_size=80,
        rnn_out_dims=[64] * 2,
        rnn_activations=functional.tanh
    )


if __name__ == '__main__':
    main()

# EOF
