import sys
import numpy as np

#every neuron has a unique connection to every previous neuron

inputs = [1, 2, 3, 2.5] #backpropogation tweaks weights # They could be inputs from the input layer or they could be the output of a previous layer.

#are these activations?

weights1 = [0.2, 0.8, 0.5, 1.0] #unique weight
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

#We're modelling 3 neurons that are receiving weights from 4 inputs from neurons from a previous layer.

#every unique neuron has a unique bias
bias1 = 2  
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)
#so this is technically the output of our neural network

