# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def multivariate_gaussian(x, mu, sigma2):
    # p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
    # density function of the examples X under the multivariate gaussian
    # distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    # treated as the covariance matrix. If Sigma2 is a vector, it is treated
    # as the \sigma^2 values of the variances in each dimension (a diagonal
    # covariance matrix)
    k = mu.size
    if len(sigma2.shape) == 1:
        sigma2 = np.diag(sigma2)
    err = x - mu
    p = (1 / (np.power(2 * np.pi, k / 2) * np.power(np.linalg.det(sigma2), 1 / 2))) \
        * np.exp((-1 / 2) * np.sum(np.dot(err, np.linalg.pinv(sigma2)) * err, axis=1))
    return p
