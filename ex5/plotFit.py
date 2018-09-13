#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import poly_features


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    x_poly = poly_features(x, p)
    for i in range(np.shape(x_poly)[0]):
        x_poly[i] = (x_poly[i] - mu) / sigma
    x_poly = np.append(np.ones((x_poly.shape[0], 1)), x_poly, axis=1)
    plt.plot(x, np.dot(x_poly, theta), '--')
