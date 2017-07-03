import torch

from torch.nn import GRUCell, Linear
from torch.nn.functional import cross_entropy

from .attention import GaussianAttention


class CategoryBranch(torch.nn.Module):
    def __init__(self, decoder_dim, output_classes, monotonic_attention=False,
                 bias=False):
        super(CategoryBranch, self).__init__()
        # TODO: Add the branch layers: CNN(s) followed by RNNs.

        self.attention = GaussianAttention(
            dim=decoder_dim, monotonic=monotonic_attention, bias=bias)
        self.decoder_cell = GRUCell(decoder_dim, decoder_dim, bias=bias)
        self.output_linear = Linear(decoder_dim, output_classes)

    def encode(self, input_):
        """Produces contexts

        Runs CNN(s) followed by RNN(s)

        """
        raise NotImplementedError()

    def get_initial_hidden(self, input_, context):
        raise NotImplementedError()

    def forward(self, input_, output):
        context = self.encode(input_)

        hidden = self.get_initial_hidden(input_, context)
        kappa = self.attention.get_inital_kappa(context)
        out_hidden = []

        out_weights = []
        for i in range(output.size(1)):
            attended, weights, kappa = self.attention(hidden, context, kappa)
            hidden = self.decoder_cell(attended, hidden)

            out_hidden.append(self.output_linear(hidden).unsqueeze(1))
            out_weights.append(weights)
        return torch.cat(out_hidden, dimension=1), weights

    def cost(self, out_hidden, target):
        out_hidden_flat = out_hidden.view(out_hidden.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        return cross_entropy(out_hidden_flat, target_flat)
