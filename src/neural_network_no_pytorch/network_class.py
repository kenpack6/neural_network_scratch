from neural_network_no_pytorch.layer_class import Layer_Dense
import numpy as np

class Network:
    
    def __init__(self, layers): #layers would be a list containing all of the layers
    
        self.layers = layers #Initialize Specified Layer architecture
        
    def forward_pass(self, input):
        """Performs the complete forward pass on all specified layers"""
        current_input = input
        
        for layer in self.layers:
            layer.forward(current_input)
            current_input = layer.output
    
        return current_input
    
    def update(self, learning_rate):
        """Updates weights and biases according to the learning rate"""
        for layer in self.layers:
            if type(layer) == Layer_Dense:
                layer.update(learning_rate)
    
    
    def back_prop(self):
        """Calculates the gradient with respect to all weights"""
        pass
   