import torch
import torch.nn as nn
import torch.nn.functional as F

#I want to copy the implementation from 3Blue1Brown, so we have 784 inputs and two hidden layers with 19 neurons and 10 outputs

class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Input Layer
        self.layer1 = nn.Linear(784,19,requires_grad_=True) #Built in module that applies a linear transformation y = xA^T + b (We pass in our data here)
        
        #Hidden Layers
        self.layer2 = nn.Linear(19,19,requires_grad_=True)
        self.layer3 = nn.Linear(19,10,requires_grad_=True)
        
        #Output
        self.layer4 = []
        
        #Softmax
        self.activation = F.softmax
        
        #loss function
        self.loss = F.relu
        
    
    def forward(self, input): #How can I access my layers that I defined?
        x = self.layer1(input)
        hidden_layers = [self.layer1, self.layer2]
        for layer in hidden_layers:
            layer(x) = x
        
        #Softmax->Loss as output
        act_x = self.activation(x)
        loss_x = self.loss(act_x)
        output = loss_x
        return output
    
    