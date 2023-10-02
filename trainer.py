import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import sys
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

#Setting up MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), #separates image into rgb channels and converts each pixel to the brightness of their colour between 0-255 which are then scaled to 0-1
                              transforms.Normalize((0.5,), (0.5,)), # normalises the tensor so it is between -1 and 1
                              ])

trainset = datasets.MNIST('./mnist/training', download=True, train=True, transform=transform)
valset = datasets.MNIST('./mnist/testing', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) #shuffles the data before each epoch during training
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

