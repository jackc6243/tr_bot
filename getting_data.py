import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def get_ordered_index(df, period_to_skip=31, val_pct=0.2, val_period=None, period_from_end_skip=0):
    """
    Get ordered index split into training and validation
    
    df: pandas dataframe
    period_to_skip: how many periods to skip at the beginning so that the indicators are accurate
    val_pct: percentage of validation data
    period_from_end_skip: 
    
    returns numpy array
    """
    # get ordered index split into training and validation
    all_dates = list(df.index)
    length = df.index[-1]
    if val_period is None:
        val_period = int(length*val_pct)
    
    val_period = length-val_period
    
    train_index = all_dates[period_to_skip:val_period]
    val_index = all_dates[val_period:]
    return train_index, val_index

def get_x(df, y_index, x_col = 'pct', period=3):
    """
    get a single series of training data
    """
    return df.loc[y_index-period:y_index-1, x_col]

def get_xy(df, period, x_col = ['pct'], y_col='pct', val_pct=0.2, val_period=None, period_to_skip=None):
    """
    return training and validation data and y_pred
    
    period: how long we want the x values to go back
    x_col: all columns of potential features
    
    """
    period_to_skip = period_to_skip if period_to_skip!=None else period+1
    
    train_index, val_index = get_ordered_index(df, period_to_skip=period_to_skip, val_pct=val_pct, val_period=val_period)

    y_train = np.array(df.loc[train_index, y_col])
    y_val = np.array(df.loc[val_index, y_col])
    
    x_train = np.zeros((len(train_index), len(x_col), period))
    x_val = np.zeros((len(val_index), len(x_col), period))
    
    for j in range(len(x_col)):
        x_column = x_col[j]

        for i, train_i in enumerate(train_index):
            temp = np.array(get_x(df, train_i, x_col = x_column, period=period))
            x_train[i, j, :] = temp

        for i, val_i in enumerate(val_index):
            temp = np.array(get_x(df, val_i, x_col = x_column, period=period))
            x_val[i, j, :] = temp

    # return np.squeeze(x_train), y_train, np.squeeze(x_val), y_val
    return x_train, np.expand_dims(y_train,1), x_val, np.expand_dims(y_val,1)


class CustomDataset(Dataset):
    """
    Custom dataset for pytorch
    """
    
    def __init__(self, Xs, y, x_transform=None, target_transform=None):
        self.Xs = Xs
        self.y = y
        self.x_transform = x_transform
        self.target_transform = target_transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        Xs = (self.Xs[i][idx].type(torch.float) for i in range(len(self.Xs)))
        y = self.y[idx].type(torch.float)

        return *Xs, y