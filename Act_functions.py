# Define our activation functions and their derivatives here

from Activation import Activation
from Layer import Layer
import numpy as np

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1-s)
        
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0,x)
        
        def relu_prime(x):
           return np.where(x > 0, 1, 0)
        