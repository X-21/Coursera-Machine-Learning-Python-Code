#!/usr/bin/env python
# -*- coding=utf-8 -*-

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from linearRegCostFunction import linear_reg_cost_function
from trainLinearReg import train_linear_reg
from learningCurve import learning_curve
from polyFeatures import poly_features
from featureNormalize import feature_normalize
from plotFit import plot_fit
from validationCurve import validation_curve


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    return loadmat(filename)


# Exercise 5 | Regularized Linear Regression and Bias-Variance
if __name__ == '__main__':

    # =========== Part 1: Loading and Visualizing Data =============
    print('Loading and Visualizing Data ...\n')
    # Load from ex5data1:
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = load_data('ex5data1.mat')
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']
    # m = Number of examples
    m = X.shape[0]
    # Plot training data
    plt.ion()
    plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.axis([-60, 40, 0, 40])
    plt.pause(2)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 2: Regularized Linear Regression Cost =============
    theta = np.array([[1], [1]])
    J, grad = linear_reg_cost_function(np.append(np.ones((m, 1)), X, axis=1), y, theta, 1)

    print('Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)\n' % J)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 3: Regularized Linear Regression Gradient =============
    print('Gradient at theta = [1 ; 1]:  [%f; %f] \n(this value should be about [-15.303016; 598.250744])\n' %
          (grad[0], grad[1]))
    print('Program paused. Press enter to continue.\n')

    # =========== Part 4: Train Linear Regression =============
    # Write Up Note: The data is non-linear, so this will not give a great fit.
    train_lambda = 0
    result = train_linear_reg(np.append(np.ones((m, 1)), X, axis=1), y, train_lambda)
    plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.plot(X, np.dot(np.append(np.ones((m, 1)), X, axis=1), result['x']), '--')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.axis([-60, 40, -10, 40])
    plt.pause(2)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 5: Learning Curve for Linear Regression =============
    curve_lambda = 0
    error_train, error_val = learning_curve(X, y, Xval, yval, curve_lambda)
    plt.figure()
    plt.plot(np.arange(m), error_train)
    plt.plot(np.arange(m), error_val)
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend(['Train', 'Cross Validation'])

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('\t%d\t\t%f\t%f\n' % (i, error_train[i], error_val[i]))

    plt.pause(2)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 6: Feature Mapping for Polynomial Regression =============
    p = 8
    # Map X onto Polynomial Features and Normalize
    X_poly = poly_features(X, p)
    # Normalize
    X_poly, mu, sigma = feature_normalize(X_poly)
    # Add Ones
    X_poly = np.append(np.ones((X_poly.shape[0], 1)), X_poly, axis=1)

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = poly_features(Xtest, p)
    for i in range(np.shape(X_poly_test)[0]):
        X_poly_test[i] = (X_poly_test[i] - mu) / sigma
    X_poly_test = np.append(np.ones((X_poly_test.shape[0], 1)), X_poly_test, axis=1)

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = poly_features(Xval, p)
    for i in range(np.shape(X_poly_val)[0]):
        X_poly_val[i] = (X_poly_val[i] - mu) / sigma
    X_poly_val = np.append(np.ones((X_poly_val.shape[0], 1)), X_poly_val, axis=1)

    print('Normalized Training Example 1:\n')
    print(X_poly[0, :], '\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 7: Learning Curve for Polynomial Regression =============
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.

    lc_pr_lambda = 1
    theta = train_linear_reg(X_poly, y, lc_pr_lambda)['x']
    # Plot training data and fit
    plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plot_fit(np.min(X), np.max(X), mu, sigma, theta, p)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = %f)' % lc_pr_lambda)
    plt.pause(2)
    plt.close()

    error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, lc_pr_lambda)
    plt.figure()
    plt.plot(np.arange(m), error_train)
    plt.plot(np.arange(m), error_val)
    plt.title('Polynomial Regression Learning Curve (lambda = %f)' % lc_pr_lambda)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend(['Train', 'Cross Validation'])

    print('Polynomial Regression (lambda = %f)\n\n' % lc_pr_lambda)
    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('\t%d\t\t%f\t%f\n' % (i, error_train[i], error_val[i]))

    plt.pause(2)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func

    # =========== Part 8: Validation for Selecting Lambda =============
    lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, yval)

    plt.figure()
    plt.plot(lambda_vec, error_train)
    plt.plot(lambda_vec, error_val)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')

    print('lambda\t\tTrain Error\tValidation Error\n')
    for i in range(len(lambda_vec)):
        print(' %f\t%f\t%f\n' % (lambda_vec[i], error_train[i], error_val[i]))

    plt.pause(2)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func
