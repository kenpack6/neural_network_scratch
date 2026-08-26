#This is to do some more practice with PyTorch as well.
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import gzip
from pathlib import Path

CURRENT_DIR = Path(__name__).resolve().parent
DATA_PATH = CURRENT_DIR.parent / "dataset" / "mnist.pkl.gz"

def load_data(data_path):
    
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
        images, labels = raw_dict[name]
        tensor_data_dict[name] = TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels,dtype=torch.long))
        
    train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
    
    validation_loader = DataLoader(validation_data,batch_size=32,shuffle=True)
    
    test_loader = DataLoader(test_data,batch_size=32,shuffle=True)
    

def main():
    load_data(DATA_PATH)

if __name__ == "__main__":
    main()