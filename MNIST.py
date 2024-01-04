# Import required libraries

import numpy as np
import pandas as pd
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from Connected import Connected
from Act_functions import Sigmoid
from MSE import mse, mse_prime
from Network import train, predict

# Use keras to import MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
# Inspect the data - Plot a sample image
sample = 7
image = X_train[sample]
# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()
print("The number shown on image is:", y_train[sample])

# Print size of arrays
print ("shape of X_train {}".format(X_train.shape))
print ("shape of X_test {}".format(X_test.shape))
print ("shape of y_train {}".format(y_train.shape))
print ("shape of y_test {}".format(y_test.shape))

# Print a sample to examine what it looks like
X_train[7] '''
'''We can see that our input is a 28x28 array of integers with values in the range 0 to 255. These correspond to the input image size 
which is 28x28 pixels and the values represent the colour intensity of each pixel in the image in the greyscale range. 0 is black, 255 
is white and the values in between are shades of grey. This can also be confirmed by the image that we have plotted above.
In order to feed the data into our NN we need to "flatten" the 28x28 array into one dimension array containing all 784 elements and 
preserving the total number of input samples. We print the resulting size.'''

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

print ("shape of X_train {}".format(X_train.shape))
#print ("shape of X_test {}".format(X_test.shape))
#X_train[7]
'''Since all our features are integer values that range from 0 to 255 it is not absolutely necessary to standardize our input data. 
However we will perform a simple scaling of the data by dividing all values by the max value 255'''
X_train = X_train/255
#print (X_train[7])
#print (y_train[7])

# Change the values of y_train and y_test to an array with binary values 
def one_hot(Y): 
  num_classes = np.max(Y) + 1

  one_hot = np.zeros((Y.size,num_classes))
  one_hot[np.arange(Y.size), Y] = 1

  return one_hot

y_train = one_hot(y_train)
y_test = one_hot(y_test)

#print ("shape of y_train {}".format(y_train.shape))

# Design the neural network
network = [
    Connected(28 * 28, 30),
    Sigmoid(),
    Connected(30,10),
    Sigmoid()    
]


