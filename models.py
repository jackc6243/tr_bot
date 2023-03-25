#data essentials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#self created tools
from getting_data import *
from Indicators import *
from y_engineering import *
from metric import *
from models import *

# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#sklearn
from sklearn.metrics import *

#visualisation with tensorboard
from torch.utils.tensorboard import SummaryWriter



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    total_loss = 0
    model.train(True)
    
    for batch, tup in enumerate(dataloader):
        
        X = tup[:-1]
        y = tup[-1]
        pred = model(*X)
        loss = loss_fn(pred, y)
        
        total_loss += loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 5 == 0:
        #   loss, current = loss.item(), batch * len(X)
        #   print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    model.train(False)
    
    print(f"Avg training loss: {total_loss/size:>8f}", end=', ')
    return total_loss/size


def test_loop(dataloader, model, loss_fn, batch_size, threshold=0.5):
    batch_num = len(dataloader.dataset)
    test_loss, correct, tot = 0, 0, 0

    with torch.no_grad():
        for tup in dataloader:
            X = tup[:-1]
            y = tup[-1]
            pred = model(*X)
            test_loss += loss_fn(pred, y).item()
            pred = (torch.sigmoid(pred) > threshold).type(torch.float32)
            correct += (pred==y).sum().item()
            tot += y.shape[0]
            
    test_loss /= batch_num
    correct /= tot
    print(f"Avg val loss: {test_loss:>8f}, Validation accuracy: {(100*correct):>0.1f}% \n")
    return test_loss, correct

class basic_indicator_cnn(nn.Module):
    def __init__(self):
        super(indicator_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv1d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, num_flat_features(self, x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x, return_logits=True, threshold = 0.5):
        temp = self.forward(x).numpy()
        logits = 1/(1 + np.exp(-temp))
        if return_logits:
            return logits
        return (l > threshold).astype("int")
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

