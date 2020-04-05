import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mnist_model import Net

n_epochs = 50
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)


use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


distilled_network = Net().to(device)
optimizer = optim.SGD(distilled_network.parameters(), lr=learning_rate,
                      momentum=momentum)
distilled_path = 'distilled_mnist_model.pth'


network = Net().to(device) 
network_state_dict = torch.load('initial_mnist_model.pth')
network.load_state_dict(network_state_dict)

distilled_network_state_dict = torch.load('distilled_mnist_model.pth', map_location={'cuda:0': 'cpu'})
distilled_network.load_state_dict(distilled_network_state_dict)

def softmax_and_cross_entropy(logits, labels):
  c,d = logits.shape
  a = -torch.bmm(logits.view(c,1,d), labels.view(c,d,1))
  a = a.view(c)
  a = torch.mean(a)
  return a

def distilled_train(epoch):
  distilled_network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data  = data.to(device)
    train_label = network(data)
    train_label = train_label/20
    train_label = F.softmax(train_label, dim = 1)
    optimizer.zero_grad()
    output = distilled_network(data)
    output = output/20
    output = F.log_softmax(output, dim = 1)
    loss = softmax_and_cross_entropy(output, train_label)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      torch.save(distilled_network.state_dict(), 'distilled_mnist_model.pth')
for epoch in range(50):
  distilled_train(epoch)

