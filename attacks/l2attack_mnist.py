import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from carlini_attack import carlini_l2_attack
import sys
sys.path.append("../mnist/")
from mnist_model import Net

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                              #  ,
                              #  torchvision.transforms.Normalize(
                              #    (0.1307,), (0.3081,))
                             ])),
  batch_size=100, shuffle=True)


network = Net().to(device)
network_state_dict = torch.load('../../distilled_mnist_model.pth', map_location={'cuda:0': 'cpu'})
# network_state_dict = torch.load('initial_model.pth')
network.load_state_dict(network_state_dict)

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
