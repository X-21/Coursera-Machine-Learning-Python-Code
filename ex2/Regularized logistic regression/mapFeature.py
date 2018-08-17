#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def map_feature(x1, x2):
    # MAPFEATURE Feature mapping function to polynomial features
    # MAPFEATURE(X1, X2) maps the two input features
    # to quadratic features used in the regularization exercise.
    # Returns a new feature array with more features, comprising of
    # X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    # Inputs X1, X2 must be the same size

    degree = 6
    x1 = x1.reshape(x1.shape[0], 1)
    x2 = x2.reshape(x2.shape[0], 1)
    out = np.ones(x1.shape)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.append(out, ((x1 ** (i - j)) * (x2 ** j)), axis=1)
    return out
