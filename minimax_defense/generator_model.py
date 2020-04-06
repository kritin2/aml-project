import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_out = 784
        self.fc0 = nn.Sequential(
                    nn.Conv2d(1,32,3,padding=1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32)
                    )
        self.fc1 = nn.Sequential(
                    nn.MaxPool2d(2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Conv2d(32,32,3,padding=1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32)
                    )
        self.fc3 = nn.Sequential(
                    nn.MaxPool2d(2)
                    )
        self.fc4 = nn.Sequential(
                    nn.Conv2d(32,32,3,padding=1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32)
                    )
        self.fc5 = nn.Sequential(
                    Upsample(scale_factor=2)
                    )
        self.fc6 = nn.Sequential(
                    nn.Conv2d(32,32,3,padding=1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32)
                    )
        self.fc7 = nn.Sequential(
                    Upsample(scale_factor=2)
                    )
        self.fc8 = nn.Sequential(
                    nn.Conv2d(32,1,3,padding=1),
                    nn.Sigmoid()
                    )
        
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
