import sys
import numpy as np
import math
np.random.seed(0)

class ReLU: 
    
    def forward(self, inputs): 
        self.inputs = inputs #this is to store the inputs for back propagation
        self.output = np.maximum(0, inputs)
        
    def backward(self):
        pass

class SoftMax:
    def forward(self, inputs): 
        self.inputs = inputs #this is to store the inputs for back propagation
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        self.output = np.exp(shifted_inputs)/np.sum(np.exp(shifted_inputs),axis=1, keepdims=True) 
    def backward(self,dout):
        #Derivative of softmax is P_i(1 - P_i) 
        P = self.output
        dot = np.sum(dout * P, axis=1, keepdims=True)
        dinputs = P * (dout - dot)
        return dinputs
        
class CrossEntropy:
    def forward(self, inputs, true_labels):
        self.true_labels = true_labels
        self.inputs = inputs 
        self.output = -np.sum(true_labels*np.log(np.clip(inputs,1e-7,1-1e-7)),axis=1, keepdims=True) # Inputs can equal 0 so we need to do np.clip (1e-7,1-1e-7) this is the range just above 0 and just below 1 which is optimal for backprop apparently
    def backward(self):
        clipped_inputs = np.clip(self.inputs,1e-7, 1 - 1e-7) # Clips input to avoid division by zero
        dinputs = -self.true_labels / clipped_inputs # Derivative of cross entropy with respect to input function
        return dinputs
            
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) #Draws from random values from a gaussian distribution
        self.biases = np.zeros((1, n_neurons)) # must take in a tuple when determining the shape
    
    def forward(self, inputs):
        """Performs the linear transformation Wx + b"""
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dout): #dout represents the gradient from the layer after it
        """Applies the chain rule locally and passes the resulting gradient to the previous layer""" 
        pass
        
    def update(self, learning_rate):
        """Updates"""
        pass
        
        
# When you save a model you really are just saving the weights and biases
#We usually initialize weights as random values between a small interval

#Input → Dense1 → ReLU → Dense2 → Softmax → Loss

# In backprop the algorithm looks like:

# Loss + Softmax → computes dout = predicted - true_labels
# Dense2 receives dout, computes dW2, db2, dinput2 → updates W2, b2, passes dinput2 back
# ReLU receives dinput2 as its dout, computes dinput3 → passes it back
# Dense1 receives that as its dout, computes dW1, db1, dinput1 → updates W1, b1

# Each layer only ever sees the gradient coming from ahead — it doesn't need to know anything about the rest of the network.
# Does seeing it traced through your actual network make it click?

#dout is the gradient from the layer ahead of the current layer

#NOTE: @ is the matrix multiplication operator, it is equivalent to np.matmul() while * represents element-wise multiplication