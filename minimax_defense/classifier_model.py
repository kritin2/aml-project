import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.n_out = 10
        self.fc0 = nn.Sequential(
                    nn.Conv2d(1,32,3,padding=1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32)
                    )
        self.fc1 = nn.Sequential(
                    nn.Conv2d(32,64,3,padding=1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(64)
                    )
        self.fc2 = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Dropout(0.25)
                    )
        self.fc3 = nn.Sequential(
                    nn.Flatten()
                    )
        self.fc4 = nn.Sequential(
                    nn.Linear(12544, 128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                    )
        self.fc5 = nn.Sequential(
                    nn.Linear(128, self.n_out),
                    nn.Softmax(dim=1)
                    )
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
