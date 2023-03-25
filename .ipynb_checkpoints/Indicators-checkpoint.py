# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:48:46 2021

data must be in pandas format with
['High', 'Low', 'Open', 'close', 'Adj close'] as columns
"""

import datetime as dt
import pandas as pd
import numpy as np

# class indicators():

def sma(data, period=20, column='close', inplace=False, col_name=False):
    """
    Simple moving average
    """
    if inplace:
        col_name = col_name if col_name else f'sma_{period}'
        data[col_name] = data[column].rolling(window=period).mean()
    return data[column].rolling(window = period).mean()


def ema(data, period=20, halflife=None, alpha=None, column='close', inplace=False, col_name=False):
    """
    Exponential moving average
    intuitively the weights of each data decreases exponentially as we go back in time
    alpha is the direct smoothing rate: 0 < alpha <=1
    """
    if inplace:
        col_name = col_name if col_name else f'ema_{period}'
        data[col_name] = data[column].ewm(
            span=period, alpha=alpha, halflife=halflife).mean()
    return data[column].ewm(span=period, alpha=alpha, halflife=halflife).mean()


def rsi(data, period=20, column='close', ema=True, inplace=False, col_name=False):
    """
    Relative strength index
    
    """
    delta = data[column].diff(1).dropna()
    avg_gain = delta.clip(lower=0).rolling(window = period).mean()
    avg_loss = delta.clip(upper=0).rolling(window = period).mean()
    
    if ema:
        avg_gain = delta.clip(lower=0).ewm(com = period - 1, adjust=True, min_periods = period).mean()
        avg_loss = delta.clip(upper=0).ewm(com = period - 1, adjust=True, min_periods = period).mean()
    else:
        avg_gain = delta.clip(lower=0).rolling(window = period).mean()
        avg_loss = delta.clip(upper=0).rolling(window = period).mean()
    
    RS = avg_gain/(abs(avg_loss))

    RSI = 100 - (100/(1+RS))
    
    if inplace:
        col_name = col_name if col_name else f'rsi_{period}'
        data[col_name] = RSI
    
    return RSI


def atr(data, smoothing='ema', period=14, column='close', high_col='high', low_col='low'):
    """
    Average True Value
    
    """
    df = data[[column, high_col, low_col]].copy()
    df['prev_close'] = df[column].shift(1)
    df['TR'] = np.maximum((df[high_col] - df[low_col]),
                          np.maximum(abs(df[high_col] - df['prev_close']),
                                     abs(df['prev_close'] - df[low_col])))

    if smoothing == 'sma':
        x = 'atr_sma'
        df[x] = sma(df, period=period, column='TR')
    elif smoothing == 'ema':
        x = 'atr_ema'
        df[x] = ema(df, period=period, column='TR')

    return df.drop(columns=[column, high_col, low_col])

def macd(data, period_long=26, period_short=12, signal=9, column='close'):
    """
    Moving Average Convergence/Divergence indicator
    
    """
    ema_long = ema(data, period_long, column=column)
    ema_short = ema(data, period_short, column=column)
    data['MACD'] = ema_short - ema_long
    data['MACD_Signal'] = ema(data, signal, column='MACD')
    
    return data[['MACD', 'MACD_Signal']]

def bol_bands(data, exponetial=True, period=20, band_mul=2, column='close', fill_na=None):
    if exponetial:
        name = f'ema_{period}'
        data[name] = ema(data, period, column=column)
        std = data[column].rolling(period).std()
        data["bollinger_up"] = data[name] + (std * band_mul)
        data["bollinger_down"] = data[name] - (std * band_mul)
        
        if fill_na != None:
            data["bollinger_up"] = data['bollinger_up'].fillna(fill_na)
            data["bollinger_down"] = data['bollinger_down'].fillna(fill_na)
        
        return data[[name, 'bollinger_up', 'bollinger_down']]
    else:
        name = f'sma_{period}'
        data[name] = sma(data, period, column=column)
        std = data[column].rolling(period).std()
        data["bollinger_up"] = data[name] + std * band_mul
        data["bollinger_down"] = data[name] - std * band_mul
        
        if fill_na != None:
            data["bollinger_up"] = data['bollinger_up'].fillna(fill_na)
            data["bollinger_down"] = data['bollinger_down'].fillna(fill_na)
            
        return data[[name, 'bollinger_up', 'bollinger_down']]


def divergence(df, look_back=5, look_front=4, period=20, divergence_col='RSI', inplace=False):
    if inplace:
        data = df.copy()

    data['local_maxima'] = 0
    data['local_minima'] = 0
    data['up_gradient'] = 0
    data['down_gradient'] = 0

    for i in range(look_back, len(data)-look_front):
        cur = data.loc[i, divergence_col]
        if data.loc[i-look_back:i+1, divergence_col].max() == cur:
            if data.loc[i:i+look_front, divergence_col].max() == cur:
                data.loc[i, "local_maxima"] = 1

        if data.loc[i-look_back:i+1, divergence_col].min() == cur:
            if data.loc[i:i+look_front, divergence_col].min() == cur:
                data.loc[i, "local_minima"] = -1

    for i in range(period, len(data)-look_front):
        x = []
        y = []

        x_ = []
        y_ = []
        for j in range(period):
            print('j:', j)
            if data.loc[i-period+j, 'local_maxima'] == 1:
                x.append(j)
                y.append(data.loc[i-period+j, divergence_col])

            if data.loc[i-period+j, 'local_minima'] == -1:
                x_.append(j)
                y_.append(data.loc[i-period+j, divergence_col])

        if len(x) > 1:
            data.loc[i, 'up_gradient'] = np.polyfit(x, y, 1)[1]

        if len(x_) > 1:
            data.loc[i, 'down_gradient'] = np.polyfit(x_, y_, 1)[1]

    if inplace:
        return data[['local_maxima', "local_minima", "up_gradient", "down_gradient"]]
