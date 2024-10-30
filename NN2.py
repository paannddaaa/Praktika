import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, stride, kernel_size, padding):
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
            stride,
    ):
        super(InceptionBlock, self).__init__()
        self.branches = []
        if out_1x1:
            self.branches.append(ConvBlock(in_channels, out_1x1, stride=1, kernel_size=1,  padding=0 ))
        self.branches.append(nn.Sequential(
            ConvBlock(in_channels, red_3x3, stride=1, kernel_size=1,  padding=0),
            ConvBlock(red_3x3, out_3x3, stride, kernel_size=3,  padding=1)))
        if (red_5x5 and out_5x5):
            self.branches.append(nn.Sequential(
                ConvBlock(in_channels, red_5x5, stride=1, kernel_size=1, padding=0),
                ConvBlock(red_5x5, out_5x5, stride, kernel_size=5,  padding=2)))
        if out_pool:
            self.branches.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, out_pool, stride=1, kernel_size=1, padding=0)))
        else:
            self.branches.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], 1)


class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=3, out_chanels=64, stride=2, kernel_size=7,  padding=5,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = nn.Sequential(
            ConvBlock(64, 64, stride=1, kernel_size=1,  padding=0),
            ConvBlock(64, 192, stride=1, kernel_size=3,  padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inseption3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32,  1)
        self.inseption3b = InceptionBlock(256, 64, 96, 128, 32, 64, 64,  1)
        self.inseption3c = InceptionBlock(320, 0, 128, 256, 32, 64, 0,  2)
        self.inseption4a = InceptionBlock(640, 256, 96, 192, 32, 64, 128, 1)
        self.inseption4b = InceptionBlock(640, 224, 112, 224, 32, 64, 128, 1)
        self.inseption4c = InceptionBlock(640, 192, 128, 256, 32, 64, 128, 1)
        self.inseption4d = InceptionBlock(640, 160, 144, 288, 32, 64, 128, 1)
        self.inseption4e = InceptionBlock(640, 0, 160, 256, 64, 128, 0, 2)
        self.inseption5a = InceptionBlock(1024, 384,192, 384, 48, 128, 128, 1)
        self.inseption5b = InceptionBlock(1024, 384, 192, 384, 48, 128, 128,1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(in_features=1024, out_features=128)
        #self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.inception2(x)
        x = F.relu(self.pool2(x))
        x = self.inseption3a(x)
        x = self.inseption3b(x)
        x = self.inseption3c(x)
        x = self.inseption4a(x)
        x = self.inseption4b(x)
        x = self.inseption4c(x)
        x = self.inseption4d(x)
        x = self.inseption4e(x)
        x = self.inseption5a(x)
        x = self.inseption5b(x)
        x = F.relu(self.avgpool1(x))
        x = self.fc(x)
        #x = F.relu(self.pool3(x))
        x = F.tanh(x)
        print(x.shape)
        return F.tanh(x)


Nelenet2 = NN2()
input_vector = torch.rand((1, 3, 220, 220))
Nelenet2(input_vector).shape
#Nelenet2.lin2.weight.shape
#summary(Nelenet2, input_size=(3, 220, 220), device="cpu")
