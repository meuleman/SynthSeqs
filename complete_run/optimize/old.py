import torch
from torch import nn


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



class tmp(nn.Module):
    def __init__(self, drop=0.2):
        super(tmp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 512, kernel_size=17, stride=1, padding=8),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(100)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.BatchNorm1d(100),
            nn.Linear(100, 16),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.net(x).view(-1, 512)
        out = self.fc_net(h)
        return out.squeeze()
