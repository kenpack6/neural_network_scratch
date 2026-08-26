#This is to do some more practice with PyTorch as well.
import torch
from neural_network_class import Neural_Network
from torch.utils.data import TensorDataset, DataLoader
import pickle
import gzip
from pathlib import Path

CURRENT_DIR = Path(__name__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "dataset" / "mnist.pkl.gz"

def load_data(data_path):
    """Loads data and return two dictionaries containing datasets and loaders"""
    with gzip.open(data_path, "r") as file:
        train_data, validation_data, test_data = pickle.load(file, encoding="latin1")
        print(type(train_data))
    
    #Convert to Tensor DataSet
    
    raw_dict = {"train": train_data,
                "validation": validation_data,
                "test": test_data}
    
    tensor_data_dict = {}
    for name, data in raw_dict.items():
        #Split labels and images
        images, labels = data
        tensor_data_dict[name] = TensorDataset(torch.tensor(images, dtype=torch.float32),
                                               torch.tensor(labels,dtype=torch.long))
        
    # Define DataLoaders
    data_loader_dict = {}
    for name, item in tensor_data_dict.items():
        loader = DataLoader(item,batch_size=32,shuffle=True)
        data_loader_dict[name] = loader
        
    return tensor_data_dict, data_loader_dict
    


def main():
    datasets, loaders = load_data(DATA_PATH)
    
    nn_model = Neural_Network()
    

if __name__ == "__main__":
    main()