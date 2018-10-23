#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

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
    # ================== Part 1: Load Example Dataset  ===================
    # We start this exercise by using a small dataset that is easy to
    # visualize.
    #
    # Our example case consists of 2 network server statistics across
    # several machines: the latency and throughput of each machine.
    # This exercise will help us find possibly faulty (or very fast) machines.

    print('Visualizing example dataset for outlier detection.\n\n')
    data = load_mat_file('./data/ex8data1.mat')
    # Visualize the example dataset
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']
    plt.ion()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker="x", s=20)
    plt.axis([0, 30, 0, 30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.pause(0.5)

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================== Part 2: Estimate the dataset statistics ===================
    # For this exercise, we assume a Gaussian distribution for the dataset.
    #
    # We first estimate the parameters of our assumed Gaussian distribution,
    # then compute the probabilities for each of the points and then visualize
    # both the overall distribution and where each of the points falls in
    # terms of that distribution.

    print('Visualizing Gaussian fit.\n\n')
    # Estimate my and sigma2
    mu, sigma2 = estimate_gaussian(X)

    p = multivariate_gaussian(X, mu, sigma2)
    # Visualize the fit
    visualize_fit(X, mu, sigma2)
    plt.pause(0.8)

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================== Part 3: Find Outliers ===================
    # Now you will find a good epsilon threshold using a cross-validation set
    # probabilities given the estimated Gaussian distribution

    pval = multivariate_gaussian(Xval, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)
    print('Best epsilon found using cross-validation: %s\n' % epsilon)
    print('Best F1 on Cross Validation Set:  %s\n' % F1)
    print('   (you should see a value epsilon of about 8.99e-05)\n')
    print('   (you should see a Best F1 value of  0.875000)\n\n')

    # Find the outliers in the training set and plot it
    outliers = np.where(p < epsilon)[0]
    plt.scatter(X[outliers, 0], X[outliers, 1], s=40, marker='o', c='', edgecolors='r')
    plt.pause(0.8)
    plt.close()

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================== Part 4: Multidimensional Outliers ===================
    # Loads the second dataset. You should now have the
    # variables X, Xval, yval in your environment
    data = load_mat_file('./data/ex8data2.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    mu, sigma2 = estimate_gaussian(X)

    p = multivariate_gaussian(X, mu, sigma2)
    pval = multivariate_gaussian(Xval, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)

    print('Best epsilon found using cross-validation: %e\n' % epsilon)
    print('Best F1 on Cross Validation Set:  %s\n' % F1)
    print('   (you should see a value epsilon of about 1.38e-18)\n')
    print('   (you should see a Best F1 value of 0.615385)\n')
    print('# Outliers found: %s\n\n' % np.sum(p < epsilon))
