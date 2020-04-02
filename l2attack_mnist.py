import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32,64, kernel_size = 3)
        self.conv4 = nn.Conv2d(64,64, kernel_size = 3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(x,2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(x,2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def carlini_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :

    images = images.to(device)     
    labels = labels.to(device)

    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images).to(device)
    w.detach_()
    w.requires_grad=True

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = (1/2*(nn.Tanh()(w) + 1)).detach()

    return attack_images


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
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                              #  ,
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=1000, shuffle=True)


import random
from torchvision import transforms, utils
from skimage import io, transform
idx = random.randint(0, 9)
    
imsize = 299
loader = transforms.Compose([

    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])
network = Net().to(device)
network_state_dict = torch.load('distilled_model.pth')
# network_state_dict = torch.load('initial_model.pth')
network.load_state_dict(network_state_dict)
from PIL import Image
import matplotlib.pyplot as plt


correct = 0
total = 0
# with torch.no_grad():
for data in test_loader:
    images, labels = data
    # print(images.shape)
    images = images.to(device)
    labels = (labels+1)%10
    labels = labels.to(device)
    images_new = carlini_l2_attack(model, images, labels , targeted = True, c = 10)
    outputs = network(images_new)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(total, correct)

print('Accuracy of the network on the 10000 test images: ', 100 * correct / total , '%' )


# Accuracy of the L2 targeted attack network on the 10000 test images:  97.49 %




# import random
# from torchvision import transforms, utils
# from skimage import io, transform
# idx = random.randint(0, 9)
    
# imsize = 299
# loader = transforms.Compose([

#     transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
# ])
# model = Net().to(device)
# # network_state_dict = torch.load('distilled_model.pth')
# network_state_dict = torch.load('initial_model.pth')
# model.load_state_dict(network_state_dict)
# from PIL import Image
# import matplotlib.pyplot as plt

# import cv2
# import numpy as np
# def image_loader(image_name):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     img_array  = np.asarray(image)
#     res = cv2.resize(img_array,(28,28))
#     gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
#     image =cv2.bitwise_not(gray)
#     # image = io.imread(image_name)
#     # image = Image.fromarray(image)
#     image = loader(image).float()
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     image = image.to(device)
#     return image  #assumes that you're using GPU
# dict_help = ['zero', 'one','two','three','four','five','six','seven','eight','nine']
# image = image_loader('mnist_complete_zero.png')
# print("originally image belongs to category : zero" ) 
# print('Trying to get ', dict_help[idx])
# image = carlini_l2_attack(model, image, torch.tensor([idx]).to(device) , targeted = True, c = 10)
# outputs = model(image)
# _, pre = torch.max(outputs.data, 1)
# print("label predicted for the image : ", dict_help[pre])
