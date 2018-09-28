#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("./Regularized logistic regression")
from mapFeature import map_feature


def plot_decision_boundary(theta, x, y):
    # Plot Data
    pos = np.where(y[:, 0] == 1.0)
    neg = np.where(y[:, 0] == 0.0)
    temp_x = x[:, [1, 2]]

    plt.ion()
    plt.figure()
    plt.scatter(temp_x[pos, 0], temp_x[pos, 1], marker="+", label="Admitted")
    plt.scatter(temp_x[neg, 0], temp_x[neg, 1], marker="o", label="Not admitted")

    m, n = x.shape
    theta = theta.reshape(theta.shape[0], 1)
    if n <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array(([x[:, 1].min(), x[:, 1].max()])).reshape(2, 1)
        plot_y = ((-1 / theta[2]) * (theta[1] * plot_x + theta[0])).reshape(2, 1)
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x[0], plot_y[0], 'rx', markersize=10)
        plt.plot(plot_x[1], plot_y[1], 'rx', markersize=10)
        plt.plot(plot_x, plot_y, '-', label="Linear regression")
        plt.axis([15, 120, 15, 120])

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(map_feature(np.array([u[i]]), np.array([v[j]])), theta)
        plt.contour(u, v, z.T, [0]).collections[0].set_label("Decision boundary")
