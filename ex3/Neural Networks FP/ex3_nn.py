#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from scipy.io import loadmat
from predict_nn_fp import predict

import sys

sys.path.append("../")
from displayData import display_data


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_mat_data(filename):
    return loadmat(filename)


if __name__ == '__main__':
    # Setup the parameters you will use for this part of the exercise
    # 20x20 Input Imc c.reshape(2,8,order='F')= a.ravel(order='F')ages of Digits
    input_layer_size = 400
    # 25 hidden units
    hidden_layer_size = 25
    # 10 labels, from 1 to 10
    num_labels = 10

    # =========== Part 1: Loading and Visualizing Data =============
    data = load_mat_data("../ex3data1.mat")
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

    # ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized neural network parameters.
    print('\nLoading Saved Neural Network Parameters ...\n')
    # Load the weights into variables Theta1 and Theta2
    theta1_2 = load_mat_data('ex3weights.mat')
    Theta1 = theta1_2['Theta1']
    Theta2 = theta1_2['Theta2']

    # ================= Part 3: Implement Predict =================
    #  After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.
    pred = predict(Theta1, Theta2, X)
    print('\nTraining Set Accuracy: \n', np.mean((pred == y) * 100))
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    rp = np.arange(0, m, 1, dtype=int)
    np.random.shuffle(rp)
    for i in range(m):
        print('\nDisplaying Example Image\n')
        display_data(X[rp[i], :].reshape(1, 400))
        pred = predict(Theta1, Theta2, X[rp[i], :].reshape(1, 400))
        print('\nNeural Network Prediction: %d (digit %d)\n' % (pred, np.mod(pred, 10)))
        print('Paused - press enter to continue, q to exit:\n')
        if input() == 'q':
            break
