import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys

X = np.loadtxt(sys.argv[1], ndmin = 2) # Read in features...
y = np.loadtxt(sys.argv[2], ndmin = 2) # ...and the labels.
p = np.loadtxt(sys.argv[3], ndmin = 2) # Read in unlabeled "predict" features

X_b = np.c_[np.ones((len(X),1)), X] # add x0 bias term to each entry in X

m = float(len(X_b)) # m is simply the number of features 

def gradient_descent(learning_rate, theta, num_iter=2000): 
    """
    Gradient descent using linear algebra.
    """
    for iteration in range(num_iter):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # calculate gradients
        theta = theta - learning_rate * gradients # redefine theta
    return theta

theta = np.random.randn(2,1) # initialize theta with random values
print 'Model Parameters for Learning Rate = 0.5'
theta_best = gradient_descent(0.5, theta, plot=False)
print 'model parameters are %s' % (theta_best)
