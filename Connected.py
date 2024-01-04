# Implement a fully connected Layer Class

from Layer import Layer
import numpy as np

class Connected(Layer):
    # Connected layer takes as arguments the number of input nodes and number of output nodes
    def __init__(self, input_size, output_size):
        # Initialise the weights and bias to random numbers
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward_pass(self, input):
        self.input = input
        # Output Y = W * X + b
        return np.dot(self.weights, self.input) + self.bias
    
    def backward_pass(self, dE_dY, learning_rate):
        # dE/dW = dE/dY * X.T
        dE_dW = np.dot(dE_dY, self.input.T)
        # dE/dX = dE/dY * W.T
        dE_dX = np.dot(dE_dY, self.weights.T)
        # update weights using learning rate
        self.weights -= learning_rate * dE_dW
        self.bias -= learning_rate * dE_dY
        return dE_dX
        