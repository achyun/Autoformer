import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, seq_len=176, dim_neck=44):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(seq_len, 2 * dim_neck, 3)
        self.conv2 = nn.Conv1d(2 * dim_neck, dim_neck, 3)
        self.conv3 = nn.Conv1d(dim_neck, dim_neck / 2, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(dim_neck)
        self.bn2 = nn.BatchNorm1d(dim_neck / 2)
        self.flatten = nn.Flatten()
        # TODO: parms
        self.dense1 = nn.Linear(1628, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        x = self.flatten(x)
        x_1 = self.dense1(x)
        true_or_false = self.sigmoid(x_1)
        return true_or_false