import torch
import torch.nn as nn
import torch.autograd 
import torch.optim 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import gzip
from pathlib import Path

#I want to copy the implementation from 3Blue1Brown, so we have 784 inputs and two hidden layers with 19 neurons and 10 outputs

class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        #NOTE: nn.Linear(batch_size, features) or nn.Linear(input_features, output_features) row x column in the layer1 case 32 images by 784 pixels

        #Input Layer
        self.layer1 = nn.Linear(784,19) #Built in module that applies a linear transformation y = xA^T + b (We pass in our data here)
        
        #Hidden Layers
        self.layer2 = nn.Linear(19,19)
        self.layer3 = nn.Linear(19,19)

        #Output Layer
        self.layer4 = nn.Linear(19,10)
        
        #Optimizer (Stochastic Gradient Descent):
        self.optimizer = torch.optim.Adam(self.parameters(),0.01)
        
        #Activation
        
        self.activation = F.relu()
        
        #Softmax
        self.softmax = F.softmax()
        
        #loss function
        self.loss = F.cross_entropy()
        

    def load_data(self, data_path):
        """Loads data and return two dictionaries containing datasets and loaders"""
        with gzip.open(data_path, "r") as file:
            train_data, validation_data, test_data = pickle.load(file, encoding="latin1")
            print(type(train_data))
        
        #Convert to Tensor DataSet
        
        raw_dict = {"train": train_data,
                    "validation": validation_data,
                    "test": test_data}
        
        tensor_data_dict = {}
        for name, data in raw_dict.items():
            #Split labels and images
            images, labels = data
            tensor_data_dict[name] = TensorDataset(torch.tensor(images, dtype=torch.float32),
                                                torch.tensor(labels,dtype=torch.long))
            
        # Define DataLoaders
        data_loader_dict = {}
        for name, item in tensor_data_dict.items():
            loader = DataLoader(item,batch_size=32,shuffle=True)
            data_loader_dict[name] = loader
            
        return tensor_data_dict, data_loader_dict

    
    def forward(self, inputs, y_labels): #How can I access my layers that I defined?
        # Input Layer
        x = self.layer1(inputs)
        
        layers = [self.activation, self.layer2, self.activation, self.layer3, self.activation]
        for layer in layers:
            x = layer(x)
            
        #Output Layer
        x = self.layer4(x)
        
        #Softmax (included in loss) -> Predictions->Loss as output
            
        loss_x = self.loss(x,y_labels)
        
        output = loss_x
        
        return output


    def training_loop(self,data_path,epochs):
        
        __, loaders = self.load_data(data_path)
        train_loader = loaders["test"]
        
        for i in range(0,epochs):
            
            print(f"Running epoch: {i+1}")
            for batch in train_loader:
                data, y_labels = batch
                loss = self.forward(data, y_labels)
                print(f"Loss({i}): {loss}")
                
                #Backward Pass
                loss.backward()
                
                #Update Gradients
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        torch.save(self.state_dict(), f"mnist_model_{epochs}.pth")
            
    