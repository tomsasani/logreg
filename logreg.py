import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression
import sys

sns.set_style('whitegrid')

X = np.loadtxt(sys.argv[1], delimiter=',') # read in features
X_b = np.c_[np.ones((len(X),1)), X] # add bias term to every feature
y = np.loadtxt(sys.argv[2], ndmin=1) # read in labels

def hyp_func(theta, x):
    """
    Define the sigmoid function. Rather than
    using linear algebra to compute theta_transpose.dot(X),
    I'm simply computing the sum of (x_j * theta_j) for 
    each feature theta_j.
    """
    z = 0
    for j in range(len(theta)):
        z += x[j] * theta[j]
    return 1 / float(1.0 + math.exp(-1 * z))

def cost_func(X, y, theta, m):
    """
    Define the cost function. Iterate over all instances,
    and calculate J(theta) by summing the cost for
    for each instance.
    """
    cost_sum = 0
    for i in range(m):
        x_i = X[i]
        h_i = float(hyp_func(theta, x_i))
        if y[i] == 1:
            error = y[i] * math.log(h_i)
        elif y[i] == 0:
            error = (1 - y[i]) * math.log(1 - h_i)
        cost_sum += error
    return (-1 / float(m)) * cost_sum

def cost_func_derivative(theta, X, y, j, m, alpha):
    """
    Define the partial derivative of the cost function,
    with respect to theta_j. 
    """
    total_error = 0
    theta = np.array(theta)
    for i in range(m):
        x_i = (X[i])
        x_i_j = (x_i[j])
        h_i = hyp_func(theta, (x_i))
        error = (h_i - y[i]) * float(x_i_j)
        total_error += error
    return (float(alpha) / float(m)) * total_error

def grad_desc(X, y, theta, m, alpha):
    """
    Implement gradient descent by looping over all
    theta_j in theta, and computing the partial differential
    of the cost function with respect to each theta_j.
    """
    new_thetas = []
    for j in range(len(theta)):
        cfd = cost_func_derivative(theta, X, y, j, m, alpha) 
        new_theta = theta[j] - cfd
        new_thetas.append(new_theta)
    return new_thetas


def log_reg(X, y, theta, alpha, cost_threshold=0.0000001):
    """
    Perform gradient descent until `change_cost` is greater
    thatn the `cost_threshold`. Redefine theta each time, 
    such that we eventually find theta (model parameters) 
    that minimize our cost function. Also, plot the cost function.
    """
    x_ax, y_ax = [], []

    m = len(y)
    change_cost = 1000 # initialize 'random' value here
    count = 0
    while change_cost > cost_threshold:
        old_cost = cost_func(X, y, theta, m) # track cost
        theta_new = grad_desc(X, y, theta, m, alpha) # get updated theta
        theta = theta_new
        new_cost = cost_func(X, y, theta_new, m) # get new, 'better' cost

        x_ax.append(count) # for plotting
        y_ax.append(new_cost) # for plotting
        count += 1

        change_cost = old_cost - new_cost
    
    plt.plot(x_ax, y_ax, c='b')
    plt.xlabel('$Iteration$')
    plt.ylabel('$Cost$')
    plt.title('$Logistic$ $Regression$ $Cost$ $Function$ $(cost threshold=1e-7)$')
    plt.savefig('cost.png')

    return theta

theta = np.array([0,0]) # initialize starting theta values
alpha = 0.01 # set alpha
theta_best = log_reg(X_b, y, theta, alpha, cost_threshold=0.00001) # get model params (best thetas)
print 'my model params = %s' % (theta_best)

def accuracy(model_params, X):
    """
    Calculate the accuracy of our model, given that
    the first and last 50 values in 'y' are 1 and 0,
    respectively.
    """
    tp = 0 
    counter = 0
    for x in X:
        pred_vals = hyp_func(model_params, x)
        if pred_vals >= 0.5:
            # y_predict = 1
            if counter < 50: # first 50 labels in y are '1'
                tp += 1
                counter += 1
            else: counter += 1
        elif pred_vals < 0.5:
            # y_predict = 0
            if counter >= 50: # last 50 labels in y are '0'
                tp += 1
                counter += 1
            else: counter += 1
    return float(tp) / counter

print 'my log_reg implementation accuracy = %s' % (accuracy(theta_best, X_b))

# Let's take a look at how Scikit-Learn performs...
logreg = LogisticRegression()
logreg.fit(X, y)
model_params_scikit = logreg.coef_
print 'scikit model params = %s' % (model_params_scikit)

y_pred_scikit = logreg.predict(X) # predict y_hat for the features in X

# Check accuracy of Scikit_Learn's predictions.
tp, counter = 0, 0
for i in y_pred_scikit:
    if i == 1 and counter < 50:
        tp += 1
        counter += 1
    elif i ==0 and counter >= 50:
        tp += 1
        counter += 1
    else:
        counter += 1

print 'scikit accuracy = %s' % (tp / float(counter))
