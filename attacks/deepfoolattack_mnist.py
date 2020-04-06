import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from carlini_attack import carlini_l2_attack
import sys
sys.path.append("../mnist/")
from mnist_model import Net

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()
                              #  ,
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                            ])),
  batch_size=100, shuffle=True)



network = Net().to(device)
network_state_dict = torch.load('../../distilled_mnist_model.pth')
# network_state_dict = torch.load('initial_model.pth')
network.load_state_dict(network_state_dict)

correct = 0
total = 0
# with torch.no_grad():
for data in test_loader:
    images, labels = data
    # print(images.shape)
    images = images.to(device)
    # labels = (labels+1)%10
    labels = labels.to(device)
    images_new = deepfool(network, images, labels , iters = 50)
    outputs = network(images_new)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted != labels).sum().item()
    print(total, correct)

print('Accuracy of the network on the 10000 test images: ', 100 * correct / total , '%' )
# Accuracy of the deepfool targeted attack network on the 10000 test images:  77.51 %

