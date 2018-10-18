# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from multivariateGaussian import multivariate_gaussian


def visualize_fit(x, mu, sigma2):
    x1, x2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    x_temp = np.vstack((np.ravel(x1), np.ravel(x2))).T
    z = multivariate_gaussian(x_temp, mu, sigma2)
    z = np.reshape(z, x1.shape)
    if not np.sum(np.isinf(z)):
        plt.contour(x1, x2, z, np.logspace(-20, -2, 7))
    return x
