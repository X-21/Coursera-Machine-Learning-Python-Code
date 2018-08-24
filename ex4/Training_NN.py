#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import scipy.optimize as sciopt
from sigmoid import sigmoid
from sigmoidGradient import sigmoid_gradient

static_grad = np.arange(0)


def training_nn_fmincg(nn_params,
                       input_layer_size,
                       hidden_layer_size,
                       num_labels,
                       x, y, nn_cost_lambda,
                       maxiter=50):
    global static_grad
    static_grad = nn_params
    print('\n')
    return sciopt.minimize(fun=cost_function, x0=nn_params,
                           args=(input_layer_size, hidden_layer_size, x, y, num_labels, nn_cost_lambda),
                           method="CG", jac=gradient, options={"maxiter": maxiter, "disp": False})


def gradient(*args):
    return static_grad.flatten()


def cost_function(nn_params,
                  input_layer_size,
                  hidden_layer_size,
                  x,
                  y,
                  num_labels,
                  nn_cost_lambda):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful variables
    m = x.shape[0]
    j = 0
    # Part 1: Feedforward the neural network and return the cost in the
    # variable J. After implementing Part 1, you can verify that your
    # cost function computation is correct by verifying the cost
    # computed in ex4.m

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    Delta1 = np.zeros(theta1.shape)
    Delta2 = np.zeros(theta2.shape)
    x = np.append(np.ones((m, 1)), x, axis=1)
    for i in range(m):
        # i-th training example info
        cur_x = x[i, :].T
        cur_y = ((np.arange(num_labels) + 1) == y[i]).T * 1
        # calculate hypothesis function (feedforward)
        a1 = cur_x
        z2 = np.dot(theta1, a1)
        a2 = np.append(np.array([1]), sigmoid(z2), axis=0)
        z3 = np.dot(theta2, a2)
        a3 = sigmoid(z3)  # = hypothesis function

        # backpropagation
        delta3 = a3 - cur_y
        delta2 = np.dot(theta2.T, delta3)[1:] * sigmoid_gradient(z2)

        Delta1 = Delta1 + np.dot(delta2.reshape((delta2.shape[0], 1)), a1.reshape((1, a1.shape[0])))
        Delta2 = Delta2 + np.dot(delta3.reshape((delta3.shape[0], 1)), a2.reshape((1, a2.shape[0])))

        # calculate the cost of this training example
        j = j + np.sum(((-cur_y) * np.log(a3)) - ((1 - cur_y) * np.log(1 - a3)))
    j = j / m
    theta1_grad = Delta1 / m
    theta2_grad = Delta2 / m
    # Regularization
    j = j + (nn_cost_lambda / (2 * m)) * \
        (np.sum(theta1[:, 1:input_layer_size + 1] ** 2) + np.sum(theta2[:, 1:hidden_layer_size + 1] ** 2))

    theta1_grad += nn_cost_lambda / m * np.append(np.zeros((theta1.shape[0], 1)), theta1[:, 1:], axis=1)
    theta2_grad += nn_cost_lambda / m * np.append(np.zeros((theta2.shape[0], 1)), theta2[:, 1:], axis=1)
    # Unroll gradients
    global static_grad
    static_grad = np.append(np.ravel(theta1_grad, order='F'), np.ravel(theta2_grad, order='F'))
    print('\rCost:%f ' % j, end='')
    return j
