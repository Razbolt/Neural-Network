# Implement the Activation Layer Class

from Layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, act_function, act_function_prime):
        self.act_function = act_function
        self.act_function_prime = act_function_prime

    # During forward pass we return the activated input as output
    def forward_pass(self, input):
        self.input = input
        self.output = self.act_function(self.input)
        return self.output
    
    # dE/dX = dE/dY * f_prime (be careful, this is an element-wise multiplication)
    def backward_pass(self, dE_dY, learning_rate):
        return np.multiply(dE_dY, self.act_function_prime(self.input))