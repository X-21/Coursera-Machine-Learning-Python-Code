#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def k_means_init_centroids(x, k):
    # Randomly reorder the indices of examples
    m = x.shape[0]
    randidx = np.random.permutation(m)
    centroids = x[randidx[0:k], :]
    return centroids
