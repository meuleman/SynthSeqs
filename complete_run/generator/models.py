# Load pytorch modules
import torch
import torch.nn as nn

from utils.net import SpectralNorm
from utils.constants import (
    SEQUENCE_LENGTH,
)


class snp_discriminator(nn.Module):
    def __init__(self,
                 num_filters,
                 len_filters,
                 pool_size,
                 fully_connected):
        super(snp_discriminator, self).__init__()

        assert SEQUENCE_LENGTH % pool_size == 0

        pool_out = SEQUENCE_LENGTH // pool_size
        self.fc_in = pool_out * num_filters

        self.conv1 = SpectralNorm(
            nn.Conv2d(1,
                      num_filters,
                      kernel_size=(len_filters, 4),
                      stride=1,
                      padding=(len_filters//2, 0),
                      bias=False)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(pool_size, 1))
        self.lrelu1 = nn.LeakyReLU(0.1, inplace=True)

        self.fc1 = SpectralNorm(nn.Linear(self.fc_in, fully_connected))
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = SpectralNorm(nn.Linear(fully_connected, 1))

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.lrelu1(h)
        h = self.fc1(h.view(-1, self.fc_in))
        h = self.lrelu2(h)
        out = self.fc2(h)
        return out 


class snp_generator(nn.Module):
    def __init__(self,
                 nz,
                 num_filters,
                 len_filters,
                 transpose_size):
        super(snp_generator, self).__init__()

        assert SEQUENCE_LENGTH % transpose_size == 0

        self.pre_transpose_len = SEQUENCE_LENGTH // transpose_size

        self.num_filters = num_filters

        self.fc = nn.Linear(nz, num_filters // 2 * self.pre_transpose_len)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(num_filters // 2)
        self.up1 = nn.ConvTranspose2d(num_filters // 2,
                                      num_filters,
                                      kernel_size=(transpose_size, 1),
                                      stride=transpose_size,
                                      bias=False)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.up2 = nn.ConvTranspose2d(num_filters,
                                      1,
                                      kernel_size=(len_filters, 4),
                                      stride=1,
                                      padding=(len_filters//2, 0))
        self.softmax = nn.Softmax(dim=3)

    def forward(self, nz):
        h = self.fc(nz).view(-1, self.num_filters//2, self.pre_transpose_len, 1)
        h = self.relu1(h)
        h = self.bn1(h)
        h = self.up1(h)
        h = self.relu2(h)
        h = self.bn2(h)
        h = self.up2(h)
        out = self.softmax(h)
        return out

