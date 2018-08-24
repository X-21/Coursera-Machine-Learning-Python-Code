# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
from debugInitializeWeights import debug_initialize_weights
from nnCostFunction import nn_cost_function
from computeNumericalGradient import compute_numerical_gradient


def check_nn_gradients(check_nn_lambda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    #  We generate some 'random' test data
    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    x = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(m) + 1, num_labels)
    # Unroll parameters
    nn_params = np.append(np.ravel(theta1, order='F'), np.ravel(theta2, order='F'))
    cost, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, check_nn_lambda)
    numgrad = compute_numerical_gradient(nn_cost_function, nn_params,
                                         input_layer_size, hidden_layer_size, num_labels, x, y, check_nn_lambda)
    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print(np.append(numgrad.reshape(numgrad.size, 1), grad.reshape(grad.size, 1), axis=1))
    print('The above two columns you get should be very similar.\n' +
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n' +
          'the relative difference will be small (less than 1e-9). \n' +
          '\nRelative Difference: %e\n' % diff)
