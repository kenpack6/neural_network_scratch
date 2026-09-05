import sys
import numpy as np
import math
np.random.seed(0)

class ReLU: #This class does not require a __init__
    
    def forward(self, inputs): 
        self.inputs = inputs #this is to store the inputs for back propagation (look into this later)
        self.output = np.maximum(0, inputs)
        
    def backward(self):
        pass

class SoftMax:
    def forward(self, inputs): 
        self.inputs = inputs #this is to store the inputs for back propagation (look into this later)
        inputs -= np.max(inputs, axis=1, keepdims= True) #ensures stability
        self.output = np.exp(inputs)/np.sum(np.exp(inputs),axis=1, keepdims=True) 
    def backward(self):
        pass
        
class CrossEntropy:
    def forward(self, inputs, true_labels):
        self.inputs = inputs  # The inputs to this function will be the probability distribution of the predicted output (after the SoftMax)
        self.output = -np.sum(true_labels*np.log(np.clip(inputs,1e-7,1-1e-7)),axis=1, keepdims=True) # Inputs can equal 0 so we need to do np.clip (1e-7,1-1e-7) this is the range just above 0 and just below 1 which is optimal for backprop apparently
    def backward(self):
        pass
            
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): 
        #When we are setting this layer we need to know the size of the input and the number of neurons.
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # we basically do the transpose here that's why the rows and columns are flipped already
        #Draws from random values from a gaussian distribution
        self.biases = np.zeros((1, n_neurons)) # must take in a tuple when determining the shape
    
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(): # dout is the gradient of the loss function with respect to the layers output 
        pass
        
    def update(self, learning_rate):
        
       self.weights = self.weights - self.dW * learning_rate
       self.biases = self.biases - self.db * learning_rate
        
        
# When you save a model you really are just saving the weights and biases

#Neural networks we usually want the interval of [-1, 1] and we want the numbers to be small.

#We usually initialize weights as random values between a small interval

#Input → Dense1 → ReLU → Dense2 → Softmax → Loss

# In backprop you go:

# Loss + Softmax → computes dout = predicted - true_labels
# Dense2 receives dout, computes dW2, db2, dinput2 → updates W2, b2, passes dinput2 back
# ReLU receives dinput2 as its dout, computes dinput3 → passes it back
# Dense1 receives that as its dout, computes dW1, db1, dinput1 → updates W1, b1

# Each layer only ever sees the gradient coming from ahead — it doesn't need to know anything about the rest of the network.
# Does seeing it traced through your actual network make it click?

#dout is the gradient from the layer ahead of the current layer