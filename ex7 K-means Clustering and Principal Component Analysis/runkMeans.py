#!/usr/bin/env python
# -*- coding=utf-8 -*-


import matplotlib.pyplot as plt
from computeCentroids import compute_centroids
from findClosestCentroids import find_closest_centroids
from plotProgresskMeans import plot_progressk_means


def run_k_means(x, initial_centroids, max_iters, plot_progress=False):
    if plot_progress:
        plt.ion()
        plt.figure()
    centroids = initial_centroids
    previous_centroids = centroids
    k = initial_centroids.shape[0]

    idx = 0

    for i in range(max_iters):
        # Output progress
        print('K-Means iteration {}/{}...\n'.format(i + 1, max_iters), flush=True)
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(x, centroids)
        # Optionally, plot progress here
        if plot_progress:
            plot_progressk_means(x, centroids, previous_centroids, idx, k, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(x, idx, k)
    if plot_progress:
        plt.close()
    return centroids, idx
