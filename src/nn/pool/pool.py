import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ChannelWiseGeM(nn.Module):
    def __init__(self, dim, p=2, eps=1e-6, requires_grad=False):
        super().__init__()
        self.ps = nn.Parameter(torch.ones(dim) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        n, C, H, W = x.shape
        batch_input = x.transpose(1, 3).reshape(n * H * W, C)
        hid = batch_input.clamp(min=self.eps).pow(self.ps)
        pooled = hid.reshape(n, H * W, C).mean(1)
        return pooled.pow(1.0 / self.ps)
