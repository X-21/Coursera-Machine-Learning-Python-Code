#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from scipy.io import loadmat
from nnCostFunction import nn_cost_function
from displayData import display_data
from sigmoidGradient import sigmoid_gradient
from randInitializeWeights import rand_initialize_weights
from contextlib import contextmanager
from checkNNGradients import check_nn_gradients
from Training_NN import training_nn_fmincg
from predict_NN import predict


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_mat_data(filename):
    return loadmat(filename)


@contextmanager
def precision_print(precision=3):
    original_options = np.get_printoptions()
    np.set_printoptions(precision=precision, suppress=True)
    try:
        yield
    finally:
        np.set_printoptions(**original_options)


if __name__ == '__main__':
    # Setup the parameters you will use for this part of the exercise
    # 20x20 Input Imc c.reshape(2,8,order='F')= a.ravel(order='F')ages of Digits
    input_layer_size = 400
    # 25 hidden units
    hidden_layer_size = 25
    # 10 labels, from 1 to 10
    num_labels = 10
    # =========== Part 1: Loading and Visualizing Data =============
    data = load_mat_data("ex4data1.mat")
    X = data['X']
    y = data['y']
    m = len(y)
    # Load Training Data
    print('Loading and Visualizing Data ...\n')
    # Randomly select 100 data points to display
    shuffle_100_X = np.arange(0, m, 1, dtype=int)
    np.random.shuffle(shuffle_100_X)
    sel = X[shuffle_100_X[0:100], :]
    display_data(sel)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================ Part 2: Loading Parameters ================
    # In this part of the exercise, we load some pre-initialized neural network parameters.
    print('\nLoading Saved Neural Network Parameters ...\n')
    # Load the weights into variables Theta1 and Theta2
    theta1_2 = load_mat_data('ex4weights.mat')
    Theta1 = theta1_2['Theta1']
    Theta2 = theta1_2['Theta2']
    nn_params = np.append(np.ravel(Theta1, order='F'), np.ravel(Theta2, order='F'), axis=0)

    # ================ Part 3: Compute Cost (Feedforward) ================
    print('\nFeedforward Using Neural Network ...\n')
    # Weight regularization parameter (we set this to 0 here).
    nn_cost_lambda = 0
    J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, nn_cost_lambda)[0]
    print('Cost at parameters (loaded from ex4weights): ', J, ' \n(this value should be about 0.287629)\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 4: Implement Regularization ===============
    # Once your cost function implementation is correct, you should now
    # continue to implement the regularization with the cost.
    print('\nChecking Cost Function (w/ Regularization) ... \n')
    # Weight regularization parameter (we set this to 1 here).
    nn_cost_lambda = 1
    J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, nn_cost_lambda)[0]
    print('Cost at parameters (loaded from ex4weights): ', J, '\n(this value should be about 0.383770)\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================ Part 5: Sigmoid Gradient  ================
    print('\nEvaluating sigmoid gradient...\n')
    g = sigmoid_gradient(np.array(([-1, -0.5, 0, 0.5, 1])))
    with precision_print(precision=6):
        print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n %s \n\n' % g)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================ Part 6: Initializing Pameters ================
    print('\nInitializing Neural Network Parameters ...\n')
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    # Unroll parameters
    initial_nn_params = np.append(np.ravel(initial_Theta1, order='F'), np.ravel(initial_Theta2, order='F'))

    # =============== Part 7: Implement Backpropagation ===============
    print('\nChecking Backpropagation... \n')
    # Check gradients by running checkNNGradients
    check_nn_gradients()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 8: Implement Regularization ===============
    print('\nChecking Backpropagation (w/ Regularization) ... \n')
    #  Check gradients by running checkNNGradients
    check_nn_lambda = 3
    check_nn_gradients(check_nn_lambda)
    # Also output the costFunction debugging values
    debug_J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, check_nn_lambda)[0]
    print('\n\nCost at (fixed) debugging parameters (w/ lambda = %d): %f ' % (check_nn_lambda, debug_J))
    print('\n(for lambda = 3, this value should be about 0.576051)\n\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =================== Part 8: Training NN ===================
    print('\nTraining Neural Network... \n')
    training_nn_fmincg_lambda = 1
    result = training_nn_fmincg(initial_nn_params, input_layer_size, hidden_layer_size, num_labels,
                                X, y, training_nn_fmincg_lambda, maxiter=50)
    theta1 = np.reshape(result['x'][0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(result['x'][hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================= Part 9: Visualize Weights =================
    print('\nVisualizing Neural Network... \n')
    display_data(theta1[:, 1:])
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================= Part 10: Implement Predict =================
    pred = predict(Theta1, Theta2, X)
    print('\nTraining Set Accuracy: \n', np.mean((pred == y) * 100))
