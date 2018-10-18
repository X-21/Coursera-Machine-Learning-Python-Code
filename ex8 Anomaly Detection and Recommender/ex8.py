#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat
from scipy.misc import imread

from estimateGaussian import estimate_gaussian
from visualizeFit import visualize_fit
from multivariateGaussian import multivariate_aussian

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
    plt.ion()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker="x", s=7)
    plt.axis([0, 30, 0, 30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.pause(0.5)
    plt.close()

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

    p = multivariate_aussian(X, mu, sigma2)
    # Visualize the fit
    visualize_fit(X,  mu, sigma2)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

    print('Program paused. Press enter to continue.\n')
    # pause_func()


