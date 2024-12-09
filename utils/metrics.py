from time import time
from math import sqrt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from .loss import LpLoss


METRIC_DICT = {
    'mae': mean_absolute_error,
    'MAE': mean_absolute_error,
    
    'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    'RMSE': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    
    'r2': r2_score,
    'R2': r2_score,
    
    'mape': mean_absolute_percentage_error,
    'MAPE': mean_absolute_percentage_error,
    
    'mse': mean_squared_error,
    'MSE': mean_squared_error,
    
    'l2': LpLoss(d=2, p=2),
    'L2': LpLoss(d=2, p=2),
}
VALID_METRICS = list(METRIC_DICT.keys())


class AverageRecord(object):
    """Computes and stores the average and current values for multidimensional data"""

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class StandardDeviationRecord:
    def __init__(self, num_features=1):
        self.num_features = num_features
        self.val = None
        self.sum = np.zeros(num_features, dtype=np.float32)
        self.sum_sq = np.zeros(num_features, dtype=np.float32)
        self.count = 0

    def update(self, val, n=None):
        val = np.array(val, dtype=np.float32)
        
        if n is None:
            n = val.shape[0]
        
        self.sum += val.sum(axis=0)
        self.sum_sq += (val ** 2).sum(axis=0)
        self.count += n

        if self.val is None:
            self.val = val
        else:
            self.val = np.concatenate([self.val, val], axis=0)

    def std(self):
        std = np.sqrt(self.sum_sq / self.count - (self.sum / self.count) ** 2)
        return std.astype(np.float32)
    
    def avg(self):
        avg = self.sum / self.count
        return avg.astype(np.float32)


class Metrics:
    def __init__(self, metrics=['mae', 'r2'], split="valid"):
        self.metric_list = metrics
        self.start_time = time()
        self.split = split
        self.metrics = {metric: AverageRecord() for metric in self.metric_list}

    def update(self, y_pred, y_true):
        for metric in self.metric_list:
            self.metrics[metric].update(METRIC_DICT[metric](y_true, y_pred))

    def compute_metrics(self):
        for metric in self.metric_list:
            self.metrics[metric] = METRIC_DICT[metric](self.y_true, self.y_pred)

    def format_metrics(self):
        result = ""
        for metric in self.metric_list:
            print(self.metrics[metric].avg)
            result += "{}: {:.8f} | ".format(metric.upper(), self.metrics[metric].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)
        
        return result

    def to_dict(self):
        return {
            metric: self.metrics[metric].avg for metric in self.metric_list
        }

    def __repr__(self):
        return self.metrics[self.metric_list[0]].avg
    
    def __str__(self):
        return self.format_metrics()


class LossRecord:
    """
    A class for keeping track of loss values during training.

    Attributes:
        start_time (float): The time when the LossRecord was created.
        loss_list (list): A list of loss names to track.
        loss_dict (dict): A dictionary mapping each loss name to an AverageRecord object.
    """

    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def format_metrics(self):
        result = ""
        for loss in self.loss_list:
            result += "{}: {:.8f} | ".format(loss, self.loss_dict[loss].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result
    
    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
