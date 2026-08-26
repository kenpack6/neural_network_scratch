#This is to do some more practice with PyTorch as well.
import torch
from neural_network_class import Neural_Network
import pickle
import gzip
from pathlib import Path

CURRENT_DIR = Path(__name__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "dataset" / "mnist.pkl.gz"

def run_model(model,epochs):        
    model.training_loop(DATA_PATH, epochs)

def test_model(model,param_path,image):
    
    model.load_state_dict(torch.load(param_path))
    with torch.no_grad():
        logits = model(image)
        prediction = torch.argmax(logits, dim=1)

def main():
    parameters = CURRENT_DIR / "mnist_model_100.pth"
    nn_model = Neural_Network()
    test_model(nn_model,parameters,)

if __name__ == "__main__":
    main()