import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mnist_model import Net


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1000, shuffle=True)

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


distilled_network = Net().to(device)
distilled_network_state_dict = torch.load('distilled_mnist_model.pth', map_location={'cuda:0': 'cpu'})
distilled_network.load_state_dict(distilled_network_state_dict)

correct = 0
total = 0
# %99.03
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = distilled_network(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: ', 100 * correct / total , '%' )
