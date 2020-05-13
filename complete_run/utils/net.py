import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


###################################################################################################################
##                                                  Spectral Norm                                                ##
## https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py ##
###################################################################################################################

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=True)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=True)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data, requires_grad=True)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length, spec_norm=False, stride=1):
        super(resnet_block, self).__init__()
        if spec_norm:
            self.net = nn.Sequential(
                SpectralNorm(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_length, stride=stride, padding=kernel_length//2, bias=False)),
                nn.ReLU(),
                SpectralNorm(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_length, stride=stride, padding=kernel_length//2, bias=False)),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_length, stride=stride, padding=kernel_length//2, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_length, stride=stride, padding=kernel_length//2, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.net(x)
        out += residual
        out = self.relu(out)
        return out


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.detach().normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.detach().normal_(1.0, 0.02)
        m.bias.detach().fill_(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
