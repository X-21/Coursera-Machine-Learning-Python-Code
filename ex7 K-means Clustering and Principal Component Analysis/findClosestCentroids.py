#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def find_closest_centroids(x, centroids):
    # Set k
    k = centroids.shape[0]

    m = x.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros((m, 1), dtype=np.int32)

    for i in range(m):
        idx[i] = 1
        min_distance = np.linalg.norm(x[i, :] - centroids[1 - 1, :]) ** 2
        for j in range(2, k + 1):
            distance = np.linalg.norm(x[i, :] - centroids[j - 1, :]) ** 2
            if distance < min_distance:
                min_distance = distance
                idx[i] = j
    return idx
