import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

def pct_log(df, y_col='close', time_to_pred = 1, pct=True, log=True):
    if log:
        y = np.log(df[y_col])
    else:
        y = df[y_col]
    
    if pct:
        return y.pct_change(time_to_pred)
    return y
    
    
def binarise(df, y_col='close', threshold=0):
    return df[y_col].pct_change().apply(lambda x: 1 if x > threshold else 0)


def put_in_bins(x, threshold):
    def thresh(a):
        if a > threshold:
            return 1
        elif a < -threshold:
            return -1
        return 0
    x = x.copy()
    for i in range(len(x)):
        x[i] = thresh(x[i])
    return x
