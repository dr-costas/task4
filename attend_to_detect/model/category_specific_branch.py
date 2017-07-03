import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import GRUCell, Linear
from torch.nn.functional import cross_entropy

from .attention import GaussianAttention


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        # TODO: proper encoder
        self.convnet = Linear(input_dim, output_dim)

    def forward(self, input_):
        input_flat = input_.view(-1, input_.size(2))
        output = self.convnet(input_flat)
        output3d = output.view(input_.size(0), input_.size(1), output.size(1))
        return output3d


class CategoryBranch(nn.Module):
    def __init__(self, input_dim, decoder_dim, output_classes,
                 monotonic_attention=False, bias=False):
        super(CategoryBranch, self).__init__()
        # TODO: Add the branch layers: CNN(s) followed by RNNs.

        self.decoder_dim = decoder_dim

        self.attention = GaussianAttention(
            dim=decoder_dim, monotonic=monotonic_attention, bias=bias)
        self.decoder_cell = GRUCell(decoder_dim, decoder_dim, bias=bias)
        self.output_linear = Linear(decoder_dim, output_classes)

        self.encoder = Encoder(input_dim, decoder_dim)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def encode(self, input_):
        """Produces contexts

        Runs CNN(s) followed by RNN(s)

        """
        return self.encoder(input_)

    def get_initial_state(self, input_, context):
        # TODO: smarter initial state
        state = Variable(torch.zeros((input_.size(0), self.decoder_dim)))
        if self.is_cuda:
            state = state.cuda()
        return state

    def forward(self, input_, output):
        context = self.encode(input_)

        hidden = self.get_initial_state(input_, context)
        kappa = self.attention.get_inital_kappa(context)

        out_hidden = []
        out_weights = []

        for i in range(output.size(1)):
            attended, weights, kappa = self.attention(hidden, context, kappa)
            hidden = self.decoder_cell(attended, hidden)

            out_hidden.append(self.output_linear(hidden).unsqueeze(1))
            out_weights.append(weights)
        return torch.cat(out_hidden, 1), out_weights

    def cost(self, out_hidden, target):
        out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
        target_flat = target.view(target.size(1))
        return cross_entropy(out_hidden_flat, target_flat)
