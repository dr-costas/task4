import torch

from torch.nn import GRUCell

from .attention import GaussianAttention


class CategoryBranch(torch.nn.Module):
    def __init__(self, decoder_dim, bias=False):
        super(CategoryBranch, self).__init__()
        # TODO: Add the branch layers: CNN(s) followed by RNNs.

        self.attention = GaussianAttention(dim=decoder_dim, bias=bias)
        self.decoder_cell = GRUCell(decoder_dim, decoder_dim, bias=bias)

    def encode(self, input_):
        """Produces contexts

        Runs CNN(s) followed by RNN(s)

        """
        raise NotImplementedError()

    def get_initial_hidden(self, input_, output):
        raise NotImplementedError()

    def forward(self, input_, output):
        context = self.encode(input_)

        hidden = self.get_initial_hidden(input_, output)
        kappa = self.attention.get_inital_kappa(context)
        out_hidden = []
        for i in range(output.size(1)):
            attended = self.attention(hidden, context, kappa)
            hidden = self.decoder_cell(attended, hidden)
            out_hidden.append(hidden)
        return out_hidden

    def cost(self, predicted, target):
        raise NotImplementedError()
