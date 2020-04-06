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
import sys
sys.path.append("../attacks/")
from carlini_attack import carlini_l2_attack


transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testset = MNIST(root='../../../', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=True)


classifier = Classifier().to(device)
discriminator = Discriminator().to(device)
generator = Generator().to(device)

discriminator.load_state_dict(torch.load('dis.pth', map_location={'cuda:0': 'cpu'}))
generator.load_state_dict(torch.load('gen.pth', map_location={'cuda:0': 'cpu'}))
classifier.load_state_dict(torch.load('class.pth', map_location={'cuda:0': 'cpu'}))

def predict(imgs):
  p = discriminator(generator(imgs))
  return p[:,:10] + p[:,10:]

def classify(predictions):
  return predictions.argmax(dim=1)

total = 0
nn_correct = 0
gan_correct = 0

for i, data in enumerate(testloader):
  imgs, labels = data

  imgs = imgs.to(device)
  labels = labels.to(device)
  label_new = ((labels+1)%10).to(device)
  imgs_new = carlini_l2_attack(classifier, imgs, label_new, targeted = True, c = 10  )
  nn_pred = classify(classifier(imgs_new))
  gan_pred = classify(predict(imgs_new))

  total += len(labels)
  nn_correct += (nn_pred == labels).sum().item()
  gan_correct += (gan_pred == labels).sum().item()
  print(total, gan_correct)


print("Training NN Accuracy: ", nn_correct/total)
print("Training Minimax Accuracy: ", gan_correct/total)


# Training NN Accuracy:  0.4851
# Training Minimax Accuracy:  0.7043