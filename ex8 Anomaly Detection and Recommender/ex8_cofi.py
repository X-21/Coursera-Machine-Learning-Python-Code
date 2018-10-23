#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from cofiCostFunc import cofi_cost_func
from checkGradients import check_nn_gradients
from estimateGaussian import estimate_gaussian
from visualizeFit import visualize_fit
from multivariateGaussian import multivariate_gaussian
from selectThreshold import select_threshold


def pause_func():
    while input() != '':
        pass


def load_mat_file(filename):
    return loadmat(filename)


if __name__ == '__main__':
    # =============== Part 1: Loading movie ratings dataset ================
    print('Loading movie ratings dataset.\n\n')
    data = load_mat_file('./data/ex8_movies.mat')
    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
    # 943 users

    # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    # rating to movie i
    R = data['R']
    Y = data['Y']
    print('Average rating for movie 1 (Toy Story): %s / 5\n\n' % np.mean(Y[0, np.where(R == 1)[1]]))

    # We can "visualize" the ratings matrix by plotting it with imagesc
    #plt.ion()
    #plt.figure()
    # plt.imshow(Y, aspect='auto')
    #plt.xlabel("Users")
    #plt.ylabel("Movies")
    # plt.pause(0.8)

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============ Part 2: Collaborative Filtering Cost Function ===========
    print('Loading movie ratings dataset.\n\n')
    data = load_mat_file('./data/ex8_movieParams.mat')
    X = data['X']
    Theta = data['Theta']
    num_users = data['num_users']
    num_movies = data['num_movies']
    num_features = data['num_features']

    # Reduce the data set size so that this runs faster
    num_users_reduce = 4
    num_movies_reduce = 5
    num_features_reduce = 3

    X_reduce = X[np.arange(num_movies_reduce), :]
    X_reduce = X_reduce[:, np.arange(num_features_reduce)]

    Theta_reduce = Theta[np.arange(num_users_reduce), :]
    Theta_reduce = Theta_reduce[:, np.arange(num_features_reduce)]

    Y_reduce = Y[np.arange(num_movies_reduce), :]
    Y_reduce = Y_reduce[:, np.arange(num_users_reduce)]

    R_reduce = R[np.arange(num_movies_reduce), :]
    R_reduce = R_reduce[:, np.arange(num_users_reduce)]

    # Evaluate cost function
    J,Grad = cofi_cost_func(np.hstack((np.ravel(X_reduce), np.ravel(Theta_reduce))), Y_reduce, R_reduce,
                       num_users_reduce, num_movies_reduce, num_features_reduce, 0)
    print('Cost at loaded parameters: %s \n(this value should be about 22.22)\n'% J)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============== Part 3: Collaborative Filtering Gradient ==============
    print('\nChecking Gradients (without regularization) ... \n')
    check_nn_gradients()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    a = 1
