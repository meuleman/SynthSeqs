from torch import nn

from utils.constants import TOTAL_CLASSES
from utils.net import resnet_block


FILTER_LENGTH = 15


class one_class(nn.Module):
    def __init__(self):
        super(one_class, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(11, 4), stride=1, padding=(5, 0), bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(kernel_size=(20, 1))
        self.fc1 = nn.Linear(1280, 200)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 1)
        self.sig = nn.Sigmoid()

    def forward(self, seq):
        h = self.conv1(seq.view(-1, 1, 100, 4))
        h = self.lrelu1(h)
        h = self.drop1(h)
        h = self.bn1(h)
        h = self.pool1(h).view(-1, 1280)
        h = self.fc1(h)
        h = self.lrelu2(h)
        h = self.drop2(h)
        h = self.fc2(h)
        out = self.sig(h).squeeze()
        return out

class resnet_one_class(nn.Module):
    def __init__(self):
        super(resnet_one_class, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 100, kernel_size=1, stride=1),
            nn.ReLU(True),
            resnet_block(100, 100, 5, spec_norm=False),
            nn.Dropout(0.5), 
            nn.MaxPool1d(20)
        )
        self.fc = nn.Linear(500, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.net(x).view(-1, 500)
        fc = self.fc(h)
        out = self.sig(fc)
        return out.squeeze()


class resnet_all_classes(nn.Module):
    def __init__(self, drop=0.2):
        super(resnet_all_classes, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 100, kernel_size=11, stride=1, padding=5),
            nn.ReLU(True),
            resnet_block(100, 100, 5, spec_norm=False),
            nn.MaxPool1d(25)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 16),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.net(x).view(-1, 400)
        out = self.fc_net(h)
        return out.squeeze()


class tmp(nn.Module):
    def __init__(self, filters, pool_out, drop=0.2):
        super(tmp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, filters, kernel_size=17, stride=1, padding=8),
            nn.ReLU(True),
            nn.BatchNorm1d(filters),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            nn.MaxPool1d(100 // pool_out)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(filters * pool_out, 100),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.BatchNorm1d(100),
            nn.Linear(100, 16),
            nn.Softmax(dim=1)
        )
        self.filters = filters
        self.pool_out = pool_out

    def forward(self, x):
        h = self.net(x.transpose(2, 3).squeeze()).view(-1, self.filters * self.pool_out)
        out = self.fc_net(h)
        return out.squeeze()


class conv_net(nn.Module):
    def __init__(self,
                 filters,
                 pool_size,
                 fully_connected,
                 drop):
        super(conv_net, self).__init__()

        filter_length = 15
        first_layer_filters, second_layer_filters = filters 
        out_length = 100 // (pool_size * pool_size)
        self.net = nn.Sequential(
            nn.Conv1d(4,
                      first_layer_filters,
                      kernel_size=filter_length,
                      stride=1,
                      padding=filter_length // 2),
            nn.ReLU(True),
            nn.BatchNorm1d(first_layer_filters),
            nn.Dropout(drop),
            nn.MaxPool1d(pool_size),

            nn.Conv1d(first_layer_filters,
                      second_layer_filters,
                      kernel_size=filter_length,
                      stride=1,
                      padding=filter_length // 2),
            nn.ReLU(True),
            nn.BatchNorm1d(second_layer_filters),
            nn.Dropout(drop),
            nn.MaxPool1d(pool_size),
        )
        self.fc_net = nn.Sequential(
            nn.Linear(out_length * second_layer_filters, fully_connected),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.BatchNorm1d(fully_connected),
            nn.Linear(fully_connected, TOTAL_CLASSES),
            nn.Softmax(dim=1)
        ) 
        self.second_layer_filters = second_layer_filters
        self.out_length = out_length

    def forward(self, x):
        h = self.net(x).view(-1, self.out_length * self.second_layer_filters)
        out = self.fc_net(h)
        return out.squeeze()
