import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.metrics import *

class anlayse_binary():
    """
    class to analyse binary results
    
    show_cf: shows and prints confusion matrix
    
    threshold_range: iterates over a range of thresholds
    
    """
    
    def __init__(self, y_pred_logits, y_true, threshold=0.5):
        self.logits = y_pred_logits
        self.y_pred = (y_pred_logits >threshold).astype("int")
        self.threshold = threshold
        self.y_true = y_true
        self.thresholds_range = None
        self.thresholds_range_x = None

        
    def binarise(self, threshold):
        self.y_pred = (self.logits > threshold).astype("int")
        self.threshold = threshold
        
    def show_cf(self):
        cf = confusion_matrix(self.y_true, self.y_pred, labels = [0,1], normalize='true')
        ConfusionMatrixDisplay(cf).plot()
        plt.show()
        print(classification_report(self.y_true, self.y_pred))
    
    def threshold_range(self, start=0.5, end=0.6, bins=10, return_ = False):
        dif = (end-start)/bins
        range_x = [start + dif*i for i in range(bins)]
        precision = []
        recall = []
        f1_score = []
        accuracy = []
        
        for i in range_x:
            y_pred = (self.logits > i).astype("int")
            cr = classification_report(self.y_true, y_pred, output_dict=True)
            precision.append(cr['1']['precision'])
            recall.append(cr['1']['recall'])
            f1_score.append(cr['1']['f1-score'])
            accuracy.append(cr['accuracy'])
        
        thresholds_range = {"precision": precision, "recall": recall, "f1-score": f1_score, "accuracy": accuracy}
        self.thresholds_range_x = range_x
        self.thresholds_range = thresholds_range
        
        if return_:
            return thresholds_range
    
    def graph_metric(self, metric):
        """
        precision, recall, f1-score, accuracy
        """
        if metric == "ROC":
            RocCurveDisplay.from_predictions(self.y_true, self.y_pred)
        
        else:
            plt.plot(self.thresholds_range_x, self.thresholds_range[metric])