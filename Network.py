# Perform the forward pass through the network
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward_pass(output)
    return output

# Create the class that is used for training the network 
def train(network, mse, mse_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += mse(y, output)

            # backward
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward_pass(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

