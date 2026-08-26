#This is to do some more practice with PyTorch as well.
import torch
from neural_network_class import Neural_Network
import pickle
import gzip
from pathlib import Path

CURRENT_DIR = Path(__name__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "dataset" / "mnist.pkl.gz"

def main():    
    nn_model = Neural_Network()
    
    nn_model.training_loop(DATA_PATH, 100)
    

if __name__ == "__main__":
    main()