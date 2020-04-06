import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mnist_model import SmallCNN
import sys
sys.path.append("../attacks/")
from carlini_attack import carlini_l2_attack

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                              #  ,
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=100, shuffle=True)


network = SmallCNN().to(device)
network_state_dict = torch.load('mnist.pth')
# network_state_dict = torch.load('initial_model.pth')
network.load_state_dict(network_state_dict)

network1 = Net().to(device)
network_state_dict = torch.load('../../distilled_mnist_model.pth')
# network_state_dict = torch.load('initial_model.pth')
network1.load_state_dict(network_state_dict)

import random
from torchvision import transforms, utils
from skimage import io, transform
idx = random.randint(0, 9)
    
imsize = 299
loader = transforms.Compose([

    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])
from PIL import Image
import matplotlib.pyplot as plt




correct = 0
total = 0
# with torch.no_grad():
for data in test_loader:
    images, labels = data
    # print(images.shape)
    images = images.to(device)

    labels_new = (labels+1)%10
    labels_new = labels_new.to(device)    
    images_new = carlini_l2_attack(network1, images, labels_new , targeted = True, c = 10)
    outputs = network(images_new)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.to(device)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(total, correct)

print('Accuracy of the network on the 10000 test images: ', 100 * correct / total , '%' )


# achieved accuracy of 95.49%
