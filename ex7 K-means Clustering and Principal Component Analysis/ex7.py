#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imread

from findClosestCentroids import find_closest_centroids
from computeCentroids import compute_centroids
from runkMeans import run_k_means
from kMeansInitCentroids import k_means_init_centroids


def pause_func():
    while input() != '':
        pass


def load_mat_file(filename):
    return loadmat(filename)


if __name__ == '__main__':
    # plt.ioff()
    # ================= Part 1: Find Closest Centroids ====================
    print('Finding closest centroids.\n\n')
    # Load an example dataset that we will be using
    data = load_mat_file('./data/ex7data2.mat')
    # Visualize the example dataset
    X = data['X']
    # Select an initial set of centroids
    K = 3  # 3 Centroids
    initial_centroids = [[3, 3], [6, 2], [8, 5]]
    initial_centroids = np.array(initial_centroids)
    # Find the closest centroids for the examples using the initial_centroids
    idx = find_closest_centroids(X, initial_centroids)
    print('Closest centroids for the first 3 examples: \n {}'.format(idx[0: 3]))
    print('\n(the closest centroids should be 1, 3, 2 respectively)\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ===================== Part 2: Compute Means =========================
    # After implementing the closest centroids function, you should now
    # complete the computeCentroids function.
    print('\nComputing centroids means.\n\n')
    #  Compute means based on the closest centroids found in the previous part.
    centroids = compute_centroids(X, idx, K)
    print('Centroids computed after initial finding of closest centroids: \n')
    print(' %s \n' % centroids)
    print('\n(the centroids should be\n')
    print('   [ 2.428301 3.157924 ]\n')
    print('   [ 5.813503 2.633656 ]\n')
    print('   [ 7.119387 3.616684 ]\n\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =================== Part 3: K-Means Clustering ======================
    # After you have completed the two functions computeCentroids and
    # findClosestCentroids, you have all the necessary pieces to run the
    # kMeans algorithm. In this part, you will run the K-Means algorithm on
    # the example dataset we have provided.

    print('\nRunning K-Means clustering on example dataset.\n\n')
    max_iters = 10
    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    centroids, idx = run_k_means(X, initial_centroids, max_iters, True)
    print('\nK-Means Done.\n\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============= Part 4: K-Means Clustering on Pixels ===============
    print('\nRunning K-Means clustering on pixels from an image.\n\n')
    A = imread('./data/bird_small.png')
    # If imread does not work for you, you can try instead
    # load_mat_file ('bird_small.mat');

    A = A / 255  # Divide by 255 so that all values are in the range 0 - 1

    # Size of the image
    img_size = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = np.reshape(A, (img_size[0] * img_size[1], 3), order='F')

    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16
    pixels_iters = 10

    # When using K-Means, it is important the initialize the centroids
    # randomly.
    # You should complete the code in kMeansInitCentroids.m before proceeding
    initial_centroids = k_means_init_centroids(X, K)

    # Run K-Means
    centroids_img, idx_img = run_k_means(X, initial_centroids, pixels_iters)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================= Part 5: Image Compression ======================
    print('\nApplying K-Means to compress an image.\n\n')

    # Find closest cluster members
    idx_img_2 = find_closest_centroids(X, centroids_img)
    X_recovered = np.zeros((idx_img_2.shape[0], X.shape[1]))
    for i in range(idx_img_2.shape[0]):
        X_recovered[i] = centroids_img[idx_img_2[i] - 1]
    X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3), order='F')

    plt.figure()
    plt.ion()
    plt.subplot(121)
    plt.imshow(A)
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(X_recovered)
    plt.title('Compressed, with {} colors.'.format(K))
    plt.pause(5)

    print('Program paused. Press enter to continue.\n')
    # pause_func()
