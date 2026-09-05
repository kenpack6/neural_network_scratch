from neural_network_no_pytorch.layer_class import Layer_Dense
import numpy as np

class Network:
    
    def __init__(self, layers): #layers would be a list containing all of the layers
        
        self.layers = layers #We keep the layers as a list
        
        
    def forward_pass(self, input):
        
        current_input = input
        for layer in self.layers:
            layer.forward(current_input)
            current_input = layer.output
            

        return current_input
    
    def update(self, learning_rate):
        for layer in self.layers:
            if type(layer) == Layer_Dense:
                layer.update(learning_rate)
    
    
    def back_prop(self, predicted, true_labels):
        
        current_dout = (predicted - true_labels) / predicted.shape[0]
        
        for layer in reversed(self.layers):
            current_dout =  layer.backward(current_dout)
            
            
            
        
        
        
        
        
        
        
    
    
    #1 Store the layers
    #2 Forward pass method
    #3 Loss function
    #4 Backprop
    #5 Adjust weights based on gradients