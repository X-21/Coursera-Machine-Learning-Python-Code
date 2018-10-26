#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from cofiCostFunc import cofi_cost_func
from checkGradients import check_gradients
from loadMovieList import load_movie_list
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
    plt.ion()
    # plt.figure()
    # plt.imshow(Y, aspect='auto')
    # plt.xlabel("Users")
    # plt.ylabel("Movies")
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
    J, Grad = cofi_cost_func(np.hstack((np.ravel(X_reduce, order='F'), np.ravel(Theta_reduce, order='F'))), Y_reduce,
                             R_reduce,
                             num_users_reduce, num_movies_reduce, num_features_reduce, 0)
    print('Cost at loaded parameters: %s \n(this value should be about 22.22)\n' % J)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============== Part 3: Collaborative Filtering Gradient ==============
    print('\nChecking Gradients (without regularization) ... \n')
    check_gradients()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ========= Part 4: Collaborative Filtering Cost Regularization ========
    J_reg, Grad_reg = cofi_cost_func(
        np.hstack((np.ravel(X_reduce, order='F'), np.ravel(Theta_reduce, order='F'))),
        Y_reduce, R_reduce,
        num_users_reduce, num_movies_reduce, num_features_reduce, 1.5
    )
    print('Cost at loaded parameters (lambda = 1.5): %s \n(this value should be about 31.34)\n' % J_reg)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ======= Part 5: Collaborative Filtering Gradient Regularization ======
    print('\nChecking Gradients (with regularization) ... \n')

    # Check gradients by running checkNNGradients
    check_gradients(1.5)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============== Part 6: Entering ratings for a new user ===============
    # Before we will train the collaborative filtering model, we will first
    # add ratings that correspond to a new user that we just observed. This
    # part of the code will also allow you to put in your own ratings for the
    # movies in our dataset!

    movieList = load_movie_list()
    # Initialize my ratings
    my_ratings = np.zeros((1682, 1))
    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings[1 - 1] = 4
    # Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings[98 - 1] = 2

    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[7 - 1] = 3
    my_ratings[12 - 1] = 5
    my_ratings[54 - 1] = 4
    my_ratings[64 - 1] = 5
    my_ratings[66 - 1] = 3
    my_ratings[69 - 1] = 5
    my_ratings[183 - 1] = 4
    my_ratings[226 - 1] = 5
    my_ratings[355 - 1] = 5

    print('\n\nNew user ratings:\n')
    for i in range(my_ratings.size):
        if my_ratings[i] > 0:
            print('Rated %.1f for %s\n' % (my_ratings[i][0], movieList[i]))

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    a = 1
