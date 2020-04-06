import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
from generator_model import Generator
from discriminator_model import Discriminator
from classifier_model import Classifier


transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])

trainset = MNIST(root='../../../', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testset = MNIST(root='../../../', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classifier = Classifier().to(device)
discriminator = Discriminator().to(device)
generator = Generator().to(device)

discriminator.load_state_dict(torch.load('dis.pth'))
generator.load_state_dict(torch.load('gen.pth'))
classifier.load_state_dict(torch.load('class.pth'))

criterion = nn.CrossEntropyLoss()
c_lr = 2e-4
d_lr = 2e-4
g_lr = 2e-4
c_optim = optim.Adadelta(classifier.parameters(), lr=c_lr, weight_decay=0.01)
d_optim = optim.Adadelta(discriminator.parameters(), lr=d_lr, weight_decay=0.01)
g_optim = optim.Adadelta(generator.parameters(), lr=g_lr, weight_decay=0.01)

def train_classifier(imgs, labels):
  c_optim.zero_grad()
  error = criterion(classifier(imgs), labels)
  error.backward()
  c_optim.step()
  return error

def train_discriminator(real_imgs, real_labels, fake_imgs, fake_labels):
  d_optim.zero_grad()
  error = criterion(discriminator(real_imgs), real_labels) + criterion(discriminator(fake_imgs), fake_labels)
  error.backward()
  d_optim.step()
  return error

def train_generator(fake_imgs, real_labels):
  g_optim.zero_grad()
  error = criterion(discriminator(fake_imgs),real_labels)
  error.backward()
  g_optim.step()
  return error



generator.train()
discriminator.train()
classifier.train()
k = 1
epochs = 50
for epoch in range(epochs):

  c_error = 0
  d_error = 0
  g_error = 0

  for i, data in enumerate(trainloader):
    imgs, labels = data

    imgs = imgs.to(device)
    labels = labels.to(device)
    fake_labels = (labels+10).to(device)
    fake_imgs = generator(imgs)
    c_error += train_classifier(imgs, labels).item()
    d_error += train_discriminator(imgs, labels, fake_imgs.detach(), fake_labels).item()
    # for j in range(k):
    g_error += train_generator(fake_imgs.detach(), labels).item()
  print('Epoch {}: c_loss: {:.8f} d_loss: {:.8f} g_loss: {:.8f}\r'.format(epoch, c_error/i, d_error/i, g_error/i))

  torch.save(classifier.state_dict(), 'class.pth')
  torch.save(generator.state_dict(), 'gen.pth')
  torch.save(discriminator.state_dict(), 'dis.pth')

# images are 1x28x28