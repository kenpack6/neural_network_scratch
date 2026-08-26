import torch
import torch.nn as nn

#I want to copy the implementation from 3Blue1Brown, so we have 784 inputs and two hidden layers with 19 neurons and 10 outputs

class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Input Layer
        self.layer1 = nn.Linear(784,19) #Built in module that applies a linear transformation y = xA^T + b (We pass in our data here)
        
        #Hidden Layers
        self.layer2 = nn.Linear(19,19)
        self.layer3 = nn.Linear(19,10)
        
        #Output
        self.layer4 = []
        
    
    def forward(data): #How can I access my layers that I defined?
        
        
        