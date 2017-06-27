from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable

from math import floor


class RNNModelWithSkipConnections(nn.Module):

    def __init__(self, n_input, n_hidden, n_readouts, n_output,
             n_filters=64, context=(21, 11)):
        super(RNNModelWithSkipConnections, self).__init__()
        n_layers = 3

        pad_t = floor((context[1] - 1)/ 2)

        self.input_layer = nn.Conv2d(
            1, n_filters,
            kernel_size=context, stride=(2, 1), padding=(0, pad_t))

        self.rnn_in_size = n_filters * floor((n_input - context[0] + 1)/2 + 1)

        self.rnn0 = nn.GRUCell(self.rnn_in_size, n_hidden, bias=False)
        self.rnn1 = nn.GRUCell(n_hidden, n_hidden, bias=False)
        self.rnn2 = nn.GRUCell(n_hidden, n_hidden, bias=False)
        self.rnns = [self.rnn0, self.rnn1, self.rnn2]

        # Transitions from input to hidden layers
        self.in_transitions1 = nn.Linear(n_input, n_hidden)
        self.in_transitions2 = nn.Linear(n_input, n_hidden)

        self.in_transitions = [self.in_transitions1, self.in_transitions2]

        self.h_transitions0 = nn.Linear(n_hidden, n_hidden)
        self.h_transitions1 = nn.Linear(n_hidden, n_hidden)
        self.h_transitions2 = nn.Linear(n_hidden, n_hidden)

        self.h_transitions = [self.h_transitions0,
                              self.h_transitions1,
                              self.h_transitions2]

        self.out_transitions0 = nn.Linear(n_hidden, n_readouts)
        self.out_transitions1 = nn.Linear(n_hidden, n_readouts)
        self.out_transitions2 = nn.Linear(n_hidden, n_readouts)

        self.out_transitions = [self.out_transitions0,
                                self.out_transitions1,
                                self.out_transitions2]

        self.output_layer = nn.Linear(n_readouts, n_output)

        self.init_weights()

        self.n_input = n_input
        self.n_filters = n_filters
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_readouts = n_readouts
        self.n_output = n_output

    def init_weights(self):
        # TODO: init other layers too, right now they're using whatever is
        # their standard init in PyTorch
        initrange = 0.1
        self.output_layer.bias.data.fill_(0)
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, h0):
        # First, do the convolutions
        #x_conv = x.transpose(0, 1).transpose(1, 2).contiguous()
        #x_conv = x_conv.view(x.size(1), 1, x.size(2), x.size(0))
        y_conv = self.input_layer(x).view(x.size(0),
                self.rnn_in_size, x.size(3))
        x_rnn = y_conv.transpose(1, 2).transpose(0, 1).contiguous()

        # Loop across timesteps
        # Only accumulate output values, the intermediate h_i are only going to be
        # used to compute the next state/output
        y = Variable(torch.zeros(x.size(3), x.size(0), x.size(2)).cuda())
        h1, h2, h3 = h0
        for t in range(x_rnn.size(0)):
            x0 = x[:, 0, :, t]
            x1 = x_rnn[t]
            x2 = self.in_transitions1.forward(x0)
            x3 = self.in_transitions2.forward(x0)

            h1 = self.rnns[0].forward(x1, h1)
            h1_to_h2 = self.h_transitions[0].forward(h1)
            h1_to_h3 = self.h_transitions[1].forward(h1)
            readouts1 = self.out_transitions[0].forward(h1)

            h2 = self.rnns[1].forward(x2 + h1_to_h2, h2)
            h2_to_h3 = self.h_transitions[2].forward(h1_to_h2)
            readouts2 = self.out_transitions[1].forward(h2)

            h3 = self.rnns[2].forward(x3 + h1_to_h3 + h2_to_h3, h3)
            readouts3 = self.out_transitions[2].forward(h3)

            # We are adding the readouts together and computing the output
            # directly based on them (no attention)
            #readouts = torch.cat([readouts1, readouts2, readouts3], 1)
            readouts = readouts1 + readouts2 + readouts3
            y[t] = self.output_layer.forward(readouts)

        return y

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = [Variable(weight.new(batch_size, self.n_hidden).zero_())
                  for _ in range(self.n_layers)]
        return hidden


def train_fn(model, optimizer, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda())
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    hidden = model.init_hidden(x.size(0))
    y_hat = model.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0]


def valid_fn(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda(), volatile=True)
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    hidden = model.init_hidden(x.size(0))
    y_hat = model.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    val_loss = criterion(y_hat, y).data[0]
    return val_loss


if __name__ == '__main__':
    x = Variable(torch.randn(5, 4, 161).cuda())
    y = Variable(torch.randn(5, 4, 161).cuda(), requires_grad=False)
    model = RNNModelWithSkipConnections(161, 20, 20, 161).cuda()
    h0 = model.init_hidden(4)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for k in range(1000):
        y_hat = model.forward(x, h0)
        loss = criterion(y_hat, y)
        print('It. {}: {}'.format(k, loss.cpu().data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

