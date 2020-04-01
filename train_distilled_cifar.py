import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
n_epochs = 100
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size= batch_size_test,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64,128, kernel_size = 3)
        self.conv4 = nn.Conv2d(128,128, kernel_size = 3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(x,2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(x,2))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


distilled_network = Net().to(device)
optimizer = optim.SGD(distilled_network.parameters(), lr=learning_rate,
                      momentum=momentum)
distilled_path = 'distilled_cifar_model.pth'


network = Net().to(device) 
network_state_dict = torch.load('initial_cifar_model.pth')
network.load_state_dict(network_state_dict)





distilled_network_state_dict = torch.load('distilled_cifar_model.pth')
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
    # print(train_label[0])
    optimizer.zero_grad()
    
    output = distilled_network(data)
    output = output/20
    output = F.log_softmax(output, dim = 1)
    # print(output[0])
    # print(train_label.shape)
    # return 1
    loss = softmax_and_cross_entropy(output, train_label)
    # print(loss.view(128,1))
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      torch.save(distilled_network.state_dict(), 'distilled_cifar_model.pth')
for epoch in range(50):
  distilled_train(epoch)





distilled_network_state_dict = torch.load('distilled_cifar_model.pth')
distilled_network.load_state_dict(distilled_network_state_dict)

correct = 0
total = 0

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




