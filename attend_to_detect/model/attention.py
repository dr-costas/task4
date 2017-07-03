import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import tanh, softmax


class ContentAttention(nn.Module):
    def __init__(self, dim):
        super(ContentAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        context_combined = torch.cat((weighted_context, input), 1)

        context_output = tanh(self.linear_out(context_combined))

        return context_output, attn


class GaussianAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super(GaussianAttention, self).__init__()
        self.linear_in = nn.Linear(dim, 2, bias=bias)
        self.linear_out = nn.Linear(dim * 2, dim, bias=bias)
        self.mask = None

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def apply_mask(self, mask):
        self.mask = mask

    def get_inital_kappa(self, context):
        batch_size = context.size(0)
        kappa = Variable(torch.zeros(batch_size))
        if self.is_cuda:
            kappa = kappa.cuda()
        return kappa

    def get_indixes(self, context):
        indexes = Variable(torch.arange(0, context.size(1)) / context.size(1),
                           requires_grad=False).unsqueeze(0)
        if self.is_cuda:
            indexes.cuda()
        return indexes

    def forward(self, input, context, kappa_prev):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        beta_kappa = torch.exp(self.linear_in(input))
        beta = beta_kappa[:, 0]
        kappa = kappa_prev + beta_kappa[:, 1]

        # Get attention
        if self.mask is not None:
            # TODO: do we need mask?
            pass
        indexes = self.get_indixes(context)
        beta_exp = beta.expand_as(indexes)
        kappa_exp = kappa.expand_as(indexes)
        attn = torch.exp(-beta_exp * (kappa_exp - indexes)**2)
        attn3 = attn.unsqueeze(2)

        weighted_context = attn3.expand_as(context) * context
        context_combined = weighted_context.sum(dim=1)

        return context_combined, attn, kappa
