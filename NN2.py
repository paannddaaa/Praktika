import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_chanels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_1x1,
            red_3x3,
            out_3x3,
            red_5x5,
            out_5x5,
            out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, stride=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1, stride=1, padding=0),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, stride=1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=3, out_chanels=64, kernel_size=7, stride=2, padding=5,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = nn.Conv2d(in_channels=64, out_channels=192, stride=1, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inseption3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inseption3b = InceptionBlock(256, 64, 96, 128, 32, 64, 64)
        self.inseption3c = InceptionBlock(320, 0, 128, 256.2, 32, 64.2, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.inception2(x)
        x = F.relu(self.pool2(x))
        x = self.inseption3a(x)
        x = self.inseption3b(x)
        x = self.inseption3c(x)
        x = F.tanh(x)
        print(x.shape)
        return F.tanh(x)


Nelenet2 = NN2()
input_vector = torch.rand((1, 3, 220, 220))
Nelenet2(input_vector).shape
#Nelenet2.lin2.weight.shape
