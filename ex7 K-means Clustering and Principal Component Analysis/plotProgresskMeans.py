#!/usr/bin/env python
# -*- coding=utf-8 -*-


import matplotlib.pyplot as plt
from plotDataPoints import plot_data_points
from drawLine import draw_line


def plot_progressk_means(x, centroids, previous, idx, k, i):
    # Plot the examples
    plot_data_points(x, idx, k)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=90, c='black', marker='x')
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous[j, :])
    plt.title('Iteration number {}'.format(i + 1))
    plt.pause(1)
