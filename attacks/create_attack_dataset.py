import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=True)

count = 0
for data in test_loader:
    images, labels = data
    a = str(labels[0].item()) + '_' + str(count) + '.png'
    print(a)
    torchvision.utils.save_image(images, a)
    print(str(labels[0]) + "fvj")
    break


