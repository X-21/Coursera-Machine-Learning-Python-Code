# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def compute_numerical_gradient(j, theta, *args):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1 = j(theta - perturb, args[0], args[1], args[2], args[3], args[4], args[5])[0]
        loss2 = j(theta + perturb, args[0], args[1], args[2], args[3], args[4], args[5])[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad
