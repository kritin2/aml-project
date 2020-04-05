import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from carlini_attack import carlini_l2_attack
import sys
sys.path.append("../cifar/")
from cifar_model import Net
from torchvision import transforms, utils



use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='../../../', train=False,
                                       download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size= 100,
                                         shuffle=False, num_workers=2)


    
imsize = 299
loader = transforms.Compose([

    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])

network = Net().to(device)
network_state_dict = torch.load('../cifar/distilled_cifar_model.pth', map_location={'cuda:0': 'cpu'} )
# network_state_dict = torch.load('initial_cifar_model.pth')
network.load_state_dict(network_state_dict)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct = 0
total = 0
# %76
for data in test_loader:
    images, labels = data
    # print(images.shape)


    images = images.to(device)
    labels = (labels+1)%10
    labels = labels.to(device)
    images_new = carlini_l2_attack(network, images, labels , targeted = True, c = 10)
    outputs = network(images_new)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(total, correct)
print('Accuracy of the network on the 10000 test images: ', 100 * correct / total , '%' )


# Accuracy of the network on the 10000 test images:  99.64 %
# import cv2
# import numpy as np
# def image_loader(image_name):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     img_array  = np.asarray(image)
#     image = cv2.resize(img_array,(32,32))
#     # image = io.imread(image_name)
#     # image = Image.fromarray(image)
#     image = loader(image).float()
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     image = image.to(device)
#     return image  #assumes that you're using GPU

# image = image_loader('bird.jpeg')
# print("originally image belongs to category : bird" ) 
# print('Trying to get ', classes[idx])
# image = carlini_l2_attack(model, image, torch.tensor([idx]).to(device) , targeted = True, c = 10)
# outputs = model(image)
# _, pre = torch.max(outputs.data, 1)
# print("label predicted for the image : ", classes[pre])
