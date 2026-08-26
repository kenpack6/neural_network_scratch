import torch
import torch.nn as nn
import torch.autograd 
import torch.optim 
import torch.nn.functional as F

#I want to copy the implementation from 3Blue1Brown, so we have 784 inputs and two hidden layers with 19 neurons and 10 outputs

class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Input Layer
        self.layer1 = nn.Linear(784,19) #Built in module that applies a linear transformation y = xA^T + b (We pass in our data here)
        
        #Hidden Layers
        self.layer2 = nn.Linear(19,19)
        self.layer3 = nn.Linear(19,19)

        #Output Layer
        self.layer4 = nn.Linear(19,10)
        
        #Optimizer (Stochastic Gradient Descent):
        self.optimizer = torch.optim.SGD()
        
        #Softmax
        self.activation = F.softmax
        
        #loss function
        self.loss = F.cross_entropy
        
    
    def forward(self, input, y_labels): #How can I access my layers that I defined?
        #Input Layer
        x = self.layer1(input)
        
        #Hidden Layers
        hidden_layers = [self.layer1, self.layer2]
        for layer in hidden_layers:
            layer(x) = x
            
        #Output Layer
        x = self.layer4(x)
        
        #Softmax->Loss as output
        
        act_x = self.activation(x)
        loss_x = self.loss(act_x,y_labels)
        output = loss_x
        
        return output
    
    def training_loop(self, data,y_labels,epochs):
        
        for i in range(0,epochs):
            print(f"Running epoch: {i}")
            loss = self.forward(self, data, y_labels)
            print(f"Loss({i}): {loss}")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
    