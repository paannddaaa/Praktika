import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 64 , kernel_size = 7, stride = 2, padding = 3, bias=True )
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride= 2, padding= 1)
        self.conv3a = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels= 192, out_channels = 384, kernel_size= 3, stride = 1, padding= 1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride= 2, padding=1)
        self.conv4a = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
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
        x = F.tanh(x)
        print(x.shape)
        return F.tanh(x)

Nelenet = NN1()
input_vector = torch.rand((1, 3, 220, 220))
Nelenet(input_vector).shape
summary(Nelenet, input_size=( 3, 220, 220), device="cpu")