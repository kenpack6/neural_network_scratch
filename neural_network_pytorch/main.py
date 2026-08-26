#This is to do some more practice with PyTorch as well.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

inputs = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]]
)

targets = torch.tensor(
    [[0.0],
     [1.0],
     [1.0],
     [0.0]]
)
