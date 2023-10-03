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
testset = datasets.MNIST('./mnist/testing', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) #shuffles the data before each epoch during training
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#Creating model
input_size = 784 #28x28 pixels is flattened to 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)) #usually used for classification problems, dim=1 specifies 2nd dimension

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(torch.cuda.is_available())
# model.to(device)

#TODO: if not trained train, otherwise load pretrained model
print("\nTraining...")
sys.stdout.flush()
criterion = nn.NLLLoss() #initialises the loss function "criterion" as the negative log likelihood loss
optimiser = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
startTime = time()
epochs = 15
for e in range(epochs):
    totalLoss = 0
    for images, labels in trainloader:
        # Flatten images into a vector of size 784
        images = images.view(images.shape[0], -1)
        #One training pass
        optimiser.zero_grad() #clears the gradients of the models parameters so they don't accumulate
        output = model(images)
        loss = criterion(output, labels)
        #Backpropagation which nudges weights and biases
        loss.backward()
        #Optimises the weights
        optimiser.step()
        
        totalLoss += loss.item()
    else:
        print("Epoch {} - Average loss: {}".format(e, totalLoss/len(trainloader)))

print("Training time (in minutes) = ",(time()-startTime)/60)
sys.stdout.flush()


#Calculate whole model's accuracy
print("\nEvaluating...")
sys.stdout.flush()
startTime = time()
correctCount, totalCount = 0, 0
for images, labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784) #reshapes to a 1D vector
    with torch.no_grad():  #no gradients are computed during forward pass as training isn't being done
        logProbs = model(img) #stores log probabilities predicted by the model for current image
    
    probabilities = torch.exp(logProbs) #converts to normal probabilities
    probabilityList = list(probabilities.numpy()[0]) #list of probabilities
    predictedLabel = probabilityList.index(max(probabilityList)) #finds index of the most probable label
    trueLabel = labels.numpy()[i]
    if(trueLabel == predictedLabel): #correct if labels match
      correctCount += 1
    totalCount += 1

print("Evaluating time (in minutes) = ", (time()-startTime)/60)
print("Number of images tested = ", totalCount)
print("Accuracy = {}%".format((correctCount/totalCount)*100))

#Saving the model
torch.save(model, './mnistModel.pt') 