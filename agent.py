import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from board import Board

# use gpu for faster operations
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# agent object
class Agent:
    
    def __init__(self):
        

        # neural network model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2*6*7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=7)
        )