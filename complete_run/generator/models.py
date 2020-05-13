# Load pytorch modules
import torch
import torch.nn as nn
import gumbel

from specnorm import SpectralNorm


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + (torch.randn(din.size()).cuda() * self.stddev)
        return din



class snp_generator_2d(nn.Module):
    def __init__(self, nz, num_filters, len_filters):
        super(snp_generator_2d, self).__init__()

        self.num_filters = num_filters

        self.fc = nn.Linear(nz, num_filters//2*10)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(num_filters//2)
        self.up1 = nn.ConvTranspose2d(num_filters//2, num_filters, (10, 1), 10, bias=False)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.up2 = nn.ConvTranspose2d(num_filters, 1, (len_filters, 4), 1, (len_filters//2, 0), bias=False)


    def forward(self, nz):
        h = self.fc(nz).view(-1, self.num_filters//2, 10, 1)
        h = self.relu1(h)
        h = self.bn1(h)
        h = self.up1(h)
        h = self.relu2(h)
        h = self.bn2(h)
        h = self.up2(h)
        output = gumbel.gumbel_softmax(h.squeeze(), 0.75)
        return output



class snp_discriminator_2d(nn.Module):
    def __init__(self):
        super(snp_discriminator_2d, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(1, 320, kernel_size=(11, 4), stride=1, padding=(5, 0), bias=False))
        self.pool1 = nn.MaxPool2d(kernel_size=(20, 1))
        self.lrelu1 = nn.LeakyReLU(0.1, inplace=True)
        self.fc1 = SpectralNorm(nn.Linear(320*5, 200))
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = SpectralNorm(nn.Linear(200, 1))

    def forward(self, x):
        h = self.conv1(x.view(-1, 1, 100, 4))
        h = self.pool1(h)
        h = self.lrelu1(h)
        h = self.fc1(h.view(-1, 320 * 5))
        h = self.lrelu2(h)
        fc = self.fc2(h)
        return fc

class snp_generator_2d_temp(nn.Module):
    def __init__(self, nz, num_filters, len_filters):
        super(snp_generator_2d_temp, self).__init__()

        self.num_filters = num_filters

        self.fc = nn.Linear(nz, num_filters//2*10)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(num_filters//2)
        self.up1 = nn.ConvTranspose2d(num_filters//2, num_filters, (10, 1), 10, bias=False)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.up2 = nn.ConvTranspose2d(num_filters, 1, (len_filters, 4), 1, (len_filters//2, 0))
        self.softmax = nn.Softmax(dim=3)

    def forward(self, nz):
        h = self.fc(nz).view(-1, self.num_filters//2, 10, 1)
        h = self.relu1(h)
        h = self.bn1(h)
        h = self.up1(h)
        h = self.relu2(h)
        h = self.bn2(h)
        h = self.up2(h)
        output = self.softmax(h)
        return output.squeeze().transpose(1, 2)


class snp_generator_2d_temp_2a(nn.Module):
    def __init__(self, nz, num_filters, len_filters):
        super(snp_generator_2d_temp_2a, self).__init__()

        self.num_filters = num_filters

        self.fc = nn.Linear(nz, num_filters//2*10)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(num_filters//2)
        self.up1 = nn.ConvTranspose2d(num_filters//2, num_filters, (10, 1), 10, bias=False)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.up2 = nn.ConvTranspose2d(num_filters, 1, (len_filters, 4), 1, (len_filters//2, 0))
        self.softmax = nn.Softmax(dim=3)

    def forward(self, nz):
        h = self.fc(nz).view(-1, self.num_filters//2, 10, 1)
        h = self.relu1(h)
        h = self.bn1(h)
        h = self.up1(h)
        h = self.relu2(h)
        h = self.bn2(h)
        h = self.up2(h)
        output = self.softmax(h)
        return output

class snp_generator_2d_temp_2b(nn.Module):
    def __init__(self, nz, num_filters, len_filters):
        super(snp_generator_2d_temp_2a, self).__init__()

        self.num_filters = num_filters

        self.fc = nn.Linear(nz, num_filters//2*10)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(num_filters//2)
        self.up1 = nn.ConvTranspose2d(num_filters//2, num_filters, (10, 1), 10, bias=False)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.up2 = nn.ConvTranspose2d(num_filters, 1, (len_filters, 4), 1, (len_filters//2, 0))
        self.softmax = nn.Softmax(dim=3)

    def forward(self, nz):
        h = self.fc(nz).view(-1, self.num_filters//2, 10, 1)
        h = self.relu1(h)
        h = self.bn1(h)
        h = self.up1(h)
        h = self.relu2(h)
        h = self.bn2(h)
        h = self.up2(h)
        output = self.softmax(h)
        return output


class resnet_generator_1d(nn.Module):
    def __init__(self, spec_norm=False):
        super(resnet_generator_1d, self).__init__()
        if spec_norm:
            self.net = nn.Sequential(
                SpectralNorm(nn.ConvTranspose1d(1, 100, kernel_size=2, stride=2, bias=False)),
                nn.ReLU(),
                resnet_block(100, 100, 5, spec_norm=True),
                #resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                SpectralNorm(nn.Conv1d(100, 4, kernel_size=1, stride=1)),
                nn.Softmax(dim=1)
            )
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose1d(1, 100, kernel_size=2, stride=2, bias=False),
                nn.ReLU(),
                resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                #resnet_block(100, 100, 5, spec_norm=False),
                nn.Conv1d(100, 4, kernel_size=1, stride=1),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        out = self.net(x.view(-1, 1, 50))
        return out


class resnet_discriminator_1d(nn.Module):
    def __init__(self, spec_norm=True):
        super(resnet_discriminator_1d, self).__init__()
        if spec_norm:
            self.net = nn.Sequential(
                GaussianNoise(0),
                SpectralNorm(nn.Conv1d(4, 100, kernel_size=1, stride=1, bias=False)),
                nn.ReLU(),
                resnet_block(100, 100, 5, spec_norm=True),
                #resnet_block(100, 100, 5, spec_norm=True),
                #resnet_block(100, 100, 5, spec_norm=True),
                #resnet_block(100, 100, 5, spec_norm=True),
                #resnet_block(100, 100, 5, spec_norm=True),
                nn.MaxPool1d(20)
            )
            self.fc_net = nn.Sequential(
                SpectralNorm(nn.Linear(500, 100)),
                nn.ReLU(),
                SpectralNorm(nn.Linear(100, 1))
            )
        else:
            self.net = nn.Sequential(
                GaussianNoise(0),
                nn.Conv1d(4, 100, kernel_size=1, stride=1),
                nn.ReLU(),
                resnet_block(100, 100, 5, spec_norm=False),
                resnet_block(100, 100, 5, spec_norm=False),
                resnet_block(100, 100, 5, spec_norm=False),
                resnet_block(100, 100, 5, spec_norm=False),
                resnet_block(100, 100, 5, spec_norm=False),
                nn.MaxPool1d(20)
            )
            self.fc = nn.Linear(500, 1)

    def forward(self, x):
        h = self.net(x).view(-1, 500)
        out = self.fc_net(h)
        return out
