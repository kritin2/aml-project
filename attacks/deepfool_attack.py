import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def deepfool(model, images, labels, iters = 100):
    images = images.to(device)
    labels = labels.to(device)
    
    for b in range(images.shape[0]) :
        
        image = images[b:b+1,:,:,:]
        label = labels[b:b+1]

        image.requires_grad = True
        output = model(image)[0]

        _, pre_0 = torch.max(output, 0)
        f_0 = output[pre_0]
        grad_f_0 = torch.autograd.grad(f_0, image, 
                                      retain_graph=False,
                                      create_graph=False)[0]
        num_classes = len(output)

        for i in range(iters):
            image.requires_grad = True
            output = model(image)[0]
            _, pre = torch.max(output, 0)
        
            if pre != pre_0 :
                image = torch.clamp(image, min=0, max=1).detach()
                break

            r = None
            min_value = None

            for k in range(num_classes) :
                if k == pre_0 :
                    continue

                f_k = output[k]
                grad_f_k = torch.autograd.grad(f_k, image, 
                                              retain_graph=True,
                                              create_graph=True)[0]

                f_prime = f_k - f_0
                grad_f_prime = grad_f_k - grad_f_0
                value = torch.abs(f_prime)/torch.norm(grad_f_prime)

                if r is None :
                    r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
                    min_value = value
                else :
                    if min_value > value :
                        r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
                        min_value = value

            image = torch.clamp(image + r, min=0, max=1).detach()

        images[b:b+1,:,:,:] = image
        
    adv_images = images
        
    return adv_images 