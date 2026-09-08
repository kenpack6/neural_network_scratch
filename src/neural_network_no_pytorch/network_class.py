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
    
    def backwards_pass(self,dout):
            """Calculates the gradient with respect to all weights"""      
               #dout is the loss function gradient
            for layer in reversed(self.layers)[1:]:
                dout = layer.backward(dout)
                
            
            
        
    def update(self, learning_rate):
        """Updates weights and biases according to the learning rate"""
        
        #Weight update
        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.weights -= learning_rate * layer.dW
                layer.biaases -= learning_rate * layer.db

   