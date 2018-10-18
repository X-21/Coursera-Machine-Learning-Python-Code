#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def compute_centroids(x, idx, k):
    # Useful variables
    m, n = x.shape
    centroids = np.zeros((k, n))
    idx = np.ravel(idx)
    for i in range(k):
        centroids[i] = np.mean(x[np.where(idx == i + 1)], axis=0)
    return centroids
