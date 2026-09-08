from neural_network_no_pytorch.layer_class import Layer_Dense, ReLU, SoftMax, CrossEntropy
import numpy as np
from neural_network_no_pytorch.network_class import Network
import gzip
from pathlib import Path
import pickle

#PATHS 
CURRENT_DIR = Path(__file__).resolve().parent
file_path = CURRENT_DIR / "dataset" / "mnist.pkl.gz"

file = gzip.open(file_path,'rb')
data = pickle.load(file, encoding="latin1")

training_data, validation_data, test_data = data #unpacks data, the data is a tuple of tuples which has label data and the actual images

training_df_images, training_df_labels = training_data #So we have 28x28 = 784. So each neuron holds an image, and the 100000 is how many images we have so (batch_size, 784) (rows can be thought as columns and columns can be thought as rows because of the ML convention to avoid doing the transpose)
valid_df_images, valid_df_labels = validation_data
test_df_images, test_df_labels = test_data


def train_run(epochs, learning_rate):
    """Runs full training loop according to number of epochs"""
    
    #Create one hot encoded labels
    one_hot_encoded_labels = np.eye(10)[training_df_labels]

    input_layer = training_df_images #Input Layer initialization

    layers = [Layer_Dense(784, 15),ReLU(),Layer_Dense(15,15),ReLU(),Layer_Dense(15, 10)] #Layer_Dense(15,10) = output layer

    neural_network = Network(layers)
    soft_max = SoftMax() #Converts logits into a probability distribution
    loss_function = CrossEntropy()

    for epoch in range(0,epochs):
        print(f"Running Epoch ({epoch})...")
        
        raw_output = neural_network.forward_pass(input_layer)
        soft_max.forward(raw_output)
        normalized = soft_max.output
        
        loss_function.forward(normalized, one_hot_encoded_labels)

        loss_func_output = loss_function.output
        print(np.mean(loss_func_output))
        
        #Calculate initial gradient from loss function:
        
        dout = loss_function.backward()
        neural_network.backwards_pass(dout)
        neural_network.update(learning_rate)
        
        acc = accuracy(neural_network, valid_df_images, valid_df_labels)
        print(f"Validation accuracy: {acc}")

def accuracy(model, inputs, labels):
    outputs = model.forward_pass(inputs)
    SoftMax.forward(outputs)
    probabilities = SoftMax.output

    predicted = np.argmax(probabilities, axis=1)
    actual = labels

    return np.mean(predicted == actual)


def main():
    train_run(10,0.01,42)

if __name__ == "__main__":
    main()