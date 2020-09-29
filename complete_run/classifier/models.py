from torch import nn

from utils.constants import TOTAL_CLASSES, SEQUENCE_LENGTH
from utils.net import resnet_block


FILTER_LENGTH = 15


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
        ) 
        self.softmax = nn.Softmax(dim=1)
        self.second_layer_filters = second_layer_filters
        self.out_length = out_length

    def forward(self, x):
        h = self.net(x).view(-1, self.out_length * self.second_layer_filters)
        fc_out = self.fc_net(h)
        softmax = self.softmax(fc_out)
        return softmax.squeeze()

    def no_softmax_forward(self, x):
        h = self.net(x).view(-1, self.out_length * self.second_layer_filters)
        fc_out = self.fc_net(h)
        return fc_out.squeeze()



class conv_net_one_layer(nn.Module):
    def __init__(self,
                 filters,
                 pool_size,
                 fully_connected,
                 drop):
        super(conv_net_one_layer, self).__init__()

        filter_length = 15
        out_length = SEQUENCE_LENGTH // pool_size
        self.net = nn.Sequential(
            nn.Conv1d(4,
                      filters,
                      kernel_size=filter_length,
                      stride=1,
                      padding=filter_length // 2),
            nn.ReLU(True),
            nn.BatchNorm1d(filters),
            nn.Dropout(drop),
            nn.MaxPool1d(pool_size),
        )
        self.fc_net = nn.Sequential(
            nn.Linear(out_length * filters, fully_connected),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.BatchNorm1d(fully_connected),
            nn.Linear(fully_connected, TOTAL_CLASSES),
        ) 
        self.softmax = nn.Softmax(dim=1)
        self.filters = filters 
        self.out_length = out_length

    def forward(self, x):
        h = self.net(x).view(-1, self.out_length * self.filters)
        fc_out = self.fc_net(h)
        softmax = self.softmax(fc_out)
        return softmax.squeeze()

    def no_softmax_forward(self, x):
        h = self.net(x).view(-1, self.out_length * self.filters)
        fc_out = self.fc_net(h)
        return fc_out.squeeze()
