import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cifar_model import Net


n_epochs = 50
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../../../', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=2)



use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

network = Net().to(device) 
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
network_path = 'initial_cifar_model.pth'


network_state_dict = torch.load('initial_cifar_model.pth' , map_location={'cuda:0': 'cpu'})
network.load_state_dict(network_state_dict)


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data  = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = network(data)
    output = output/20
    output = F.log_softmax(output, dim = 1)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      torch.save(network.state_dict(), 'initial_cifar_model.pth')
for epoch in range(1, n_epochs + 1):
  train(epoch)

