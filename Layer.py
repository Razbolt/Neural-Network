# Define our parent Layer class here

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input):
        #return the output
        pass

    def backward_pass(self, output_gradient, learning_rate):
        #return the input gradient
        pass