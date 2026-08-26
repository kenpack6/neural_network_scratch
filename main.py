from layer_class import Layer_Dense, ReLU, SoftMax, CrossEntropy
import numpy as np
from network_class import Network
import gzip
import pickle

file_path = "C:/Users/kenne/Desktop/ML Research - Jake Rhodes/Neural Network Practice/my_first_neural_network/dataset/mnist.pkl.gz" #r ensures that it doesn't see backslashes as starting a new line

file = gzip.open(file_path,'rb')
data = pickle.load(file, encoding="latin1")

training_data, validation_data, test_data = data #unpacks data, the data is a tuple of tuples which has label data and the actual images

training_df_images, training_df_labels = training_data #So we have 28x28 = 784. So each neuron holds an image, and the 100000 is how many images we have so (batch_size, 784) (rows can be thought as columns and columns can be thought as rows because of the ML convention to avoid doing the transpose)
valid_df_images, valid_df_labels = validation_data
test_df_images, test_df_labels = test_data

# print(training_df_labels)
one_hot_encoded_labels = np.eye(10)[training_df_labels]

# print(one_hot_encoded_labels)

#Okay, how do neural networks work? We have a forward pass that takes in input then the hidden layers then we have an output that includes a loss function then we do backpropogation to fix the weights and biases

#So now I'm going to make a neural network

# print(type(training_df_images))

input_layer = training_df_images

layers = [Layer_Dense(784, 15),ReLU(),Layer_Dense(15,15),ReLU(),Layer_Dense(15, 10)]

neural_network = Network(layers)
soft_max = SoftMax()
loss_function = CrossEntropy()

#TRAINING LOOP

epochs = 0

for epoch in range(0,epochs):
    raw_output = neural_network.forward_pass(input_layer)
    soft_max.forward(raw_output)
    normalized = soft_max.output
    
    loss_function.forward(normalized, one_hot_encoded_labels)

    loss_func_output = loss_function.output
    print(np.mean(loss_func_output))

    # print(loss_func_output)

    neural_network.back_prop(normalized, one_hot_encoded_labels)

    neural_network.update(1)
    


# print(np.mean(loss_func_output))


#COMPLETE FORWARD PASS

# 1 One-hot encode your labels

# So the way this would work is the cross entropy function is taking as input a probability distribution and finding how much the predicted distribution is off from the actual distribution. The actual distribution is just everything else is 0 except the actual label.

# 2 Add CrossEntropy to your network or training loop
# 3 Implement backpropagation
# 4 Implement the training loop