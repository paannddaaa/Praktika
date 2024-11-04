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

class InceptionBlockFull(nn.Module):
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
        super(InceptionBlockFull, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, stride=1, kernel_size=1,  padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, stride=1, kernel_size=1,  padding=0),
            ConvBlock(red_3x3, out_3x3, stride, kernel_size=3,  padding=1)
            )
        self.branch3 = nn.Sequential(
                ConvBlock(in_channels, red_5x5, stride=1, kernel_size=1, padding=0),
                ConvBlock(red_5x5, out_5x5, stride, kernel_size=5,  padding=2)
                )
        if out_pool:
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, out_pool, stride=1, kernel_size=1, padding=0)
                )
        else:
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class InceptionBlockTransition(nn.Module):
    def __init__(
            self,
            in_channels,
            red_3x3,
            out_3x3,
            red_5x5,
            out_5x5,
            out_pool,
            stride,
    ):
        super(InceptionBlockTransition, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, stride=1, kernel_size=1,  padding=0),
            ConvBlock(red_3x3, out_3x3, stride, kernel_size=3,  padding=1)
            )
        self.branch2 = nn.Sequential(
                ConvBlock(in_channels, red_5x5, stride=1, kernel_size=1, padding=0),
                ConvBlock(red_5x5, out_5x5, stride, kernel_size=5,  padding=2)
                )
        if out_pool:
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, out_pool, stride=1, kernel_size=1, padding=0)
                )
        else:
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], 1)

class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 64 , kernel_size = 7, stride = 2, padding = 3)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride= 2, padding= 1)
        self.conv3a = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels= 192, out_channels = 384, kernel_size= 3, stride = 1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride= 2, padding=1)
        self.conv4a = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features = 12544, out_features =8192 )
        self.fc2 = nn.Linear(in_features = 8192, out_features = 4096)
        self.fc7128 = nn.Linear(in_features= 4096, out_features=128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4(x)
        x = self.conv5a(x)
        x = self.conv5(x)
        x = self.conv6a(x)
        x = self.conv6(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc7128(x)

        return x


class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_chanels=64, stride=2, kernel_size=7,  padding=5,)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception2 = nn.Sequential(
            ConvBlock(64, 64, stride=1, kernel_size=1,  padding=0),
            ConvBlock(64, 192, stride=1, kernel_size=3,  padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlockFull(192, 64, 96, 128, 16, 32, 32,  1)
        self.inception3b = InceptionBlockFull(256, 64, 96, 128, 32, 64, 64,  1)
        self.inception3c = InceptionBlockTransition(320, 128, 256, 32, 64, 0,  2)
        self.inception4a = InceptionBlockFull(640, 256, 96, 192, 32, 64, 128, 1)
        self.inception4b = InceptionBlockFull(640, 224, 112, 224, 32, 64, 128, 1)
        self.inception4c = InceptionBlockFull(640, 192, 128, 256, 32, 64, 128, 1)
        self.inception4d = InceptionBlockFull(640, 160, 144, 288, 32, 64, 128, 1)
        self.inception4e = InceptionBlockTransition(640, 160, 256, 64, 128, 0, 2)
        self.inception5a = InceptionBlockFull(1024, 384,192, 384, 48, 128, 128, 1)
        self.inception5b = InceptionBlockFull(1024, 384, 192, 384, 48, 128, 128,1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(in_features=1024, out_features=128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.inception2(x)
        x = F.relu(self.pool2(x))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = F.relu(self.avgpool1(x))
        x = x.reshape((-1, 1024))
        x = self.fc(x)

        return x

    def test():
        from torchsummary import summary
        Nelenet2 = NN1()
        summary(Nelenet2, (3, 220, 220), device='cpu')
        Nelenet2 = NN2()
        summary(Nelenet2, (3, 220, 220), device='cpu')

    if __name__ == '__main__':
        test()