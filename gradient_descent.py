import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys

sns.set_style('whitegrid')

X = np.loadtxt(sys.argv[1], ndmin = 2) # Read in features...
y = np.loadtxt(sys.argv[2], ndmin = 2) # ...and the labels.
p = np.loadtxt(sys.argv[3], ndmin = 2) # Read in unlabeled "predict" features

X_b = np.c_[np.ones((len(X),1)), X] # add x0 bias term to each entry in X

m = float(len(X_b)) # m is simply the number of features 

X_new = np.array([[np.min(X)],[np.max(X)]]) # reduce dimensions for plotting
X_new_b = np.c_[np.ones((len(X_new),1)), X_new] # add bias term to each entry

def gradient_descent(learning_rate, theta, num_iter=2000, 
                        print_models=True, return_theta_best=False, plot=True):
    plt.scatter(X, y) # plot the original data
    for iteration in range(num_iter):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # calculate gradients
        theta = theta - learning_rate * gradients # redefine theta
        if plot: 
            if iteration % 200 == 0:
                plt.plot(X_new, X_new_b.dot(theta)) # plot line of best fit
        if print_models: 
            if iteration in range(1, 6) or iteration == num_iter - 1:
                print theta
        if return_theta_best: 
            if iteration == num_iter - 1:
                return theta
    if plot:
        plt.xlabel('$X$ $(feature)$')
        plt.ylabel('$y$ $(labels)$')
        plt.title('$Linear$ $Regression$ $with$ $Gradient$ $Descent$')
        plt.savefig('%s.png' % (learning_rate))

"""
Report model parameters for the first five, as well
as the last, iteration for each learning rate.
"""
theta = np.random.randn(2,1) # initialize theta with random values
print 'Model Parameters for Learning Rate = 0.5'
gradient_descent(0.5, theta, plot=False)
print 'Model Parameters for Learning Rate = 0.01'
gradient_descent(0.01, theta, plot=False)
print 'Model Parameters for Learning Rate = 0.0001'
gradient_descent(0.0001, theta, plot=True)

"""
Predict values using the theta value obtained after 2000 iterations
with a learning rate of 0.01.
"""
theta_best = gradient_descent(0.01, theta, print_models=False, 
                                return_theta_best=True, plot=False) # theta from 2000th iteration of gradient descent

p_b = np.c_[np.ones((len(p),1)), p] # add x0 bias term to each 'predict' entry

y_predict = []
for instance in p_b:
    y_predict.append(theta_best.T.dot(instance))
print 'Predicted values from batch gradient descent:'
print y_predict

"""
Model the data with SciKit-Learn's version of linear regression.
"""
lin_reg = LinearRegression() 
lin_reg.fit(X, y)
print 'Model Parameters from SKL linear regression.'
print lin_reg.intercept_, lin_reg.coef_

"""
Predict values using model parameters from SKL.
"""
y_predict_skl = []
for instance in p:
    y_predict_skl.append(lin_reg.predict(instance))
print 'Predicted values from SKL:'
print y_predict_skl
