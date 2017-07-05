import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import GRUCell, Linear
from torch.nn.functional import cross_entropy

from .attention import GaussianAttention


class Encoder(nn.Module):
    def __init__(self, input_dim, n_filters, kernel_shape=(3,3), stride=(2,2)):
        super(Encoder, self).__init__()
        self.convnet = nn.Conv2d(
            input_dim[0], n_filters,
            kernel_size=kernel_shape,
            stride=stride,
            padding=((kernel_shape[0]-1)//2, (kernel_shape[1]-1)//2)
        )
        self.n_filters = n_filters
        self.input_dim = input_dim
        self.output_dim = (input_dim[1]//stride[0], n_filters * input_dim[2]//stride[1])

    def forward(self, x):
        '''input shape: (batch_size, 1, n_bins, n_frames)
        output shape: (batch_size, n_frames/stride[1], n_filters * n_bins/stride[0])
        '''
        output3d = self.convnet(x).transpose(1, 2).contiguous()
        # shape is (batch_size, n_filters, n_frames/stride[0], n_bins/stride[1])
        output = output3d.view(output3d.size(0), output3d.size(1), -1)
        return output


class CategoryBranch(nn.Module):
    """One category branch

    Intended use:
    >>> branch = CategoryBranch(5, 5, 3)
    >>> hid, weights = branch(input_, output)
    >>> cost = branch.cost(hid, output)

    """
    def __init__(self, input_dim, decoder_dim, output_classes,
            enc_filters=64, enc_stride=(2, 2), enc_kernel_shape=(3,3),
            monotonic_attention=False, bias=False):
        super(CategoryBranch, self).__init__()

        self.out = None

        self.encoder = Encoder(input_dim, enc_filters, enc_kernel_shape, enc_stride)

        self.decoder_dim = decoder_dim
        self.attention = GaussianAttention(
            dim=self.decoder_dim, monotonic=monotonic_attention, bias=bias)
        self.decoder_cell = GRUCell(self.encoder.output_dim[1], self.decoder_dim, bias=bias)
        self.output_linear = Linear(decoder_dim, output_classes)


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

    def forward(self, input_, output_len):
        context = self.encode(input_)

        hidden = self.get_initial_state(input_, context)
        kappa = self.attention.get_initial_kappa(context)

        out_hidden = []
        out_weights = []

        for i in range(output_len):
            attended, weights, kappa = self.attention(hidden, context, kappa)
            hidden = self.decoder_cell(attended, hidden)

            out_hidden.append(self.output_linear(hidden).unsqueeze(1))
            out_weights.append(weights)
        return torch.cat(out_hidden, 1), out_weights

    @staticmethod
    def cost(self, out_hidden, target):
        out_hidden_flat = out_hidden.view(-1, out_hidden.size(2))
        target_flat = target.view(target.size(1))
        return cross_entropy(out_hidden_flat, target_flat)


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    branch = CategoryBranch((100, 10), 32, 10)
    x = Variable(torch.rand(16, 1, 100, 10))

    pred = branch(x, 3)
    print(pred)
