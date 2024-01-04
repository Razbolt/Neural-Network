# Define the MSE calculation and its derivative

import numpy as np

# E = 1/n Sum(y_pred - y_true)^2
def mse(y_pred, y_true):
    return np.mean((np.array(y_pred) - np.array(y_true)) **2)

# dE/dY = 2/n(y_pred - y_true)
def mse_prime(y_pred, y_true):
    return 2 * (y_pred - y_true)/np.size(y_true)