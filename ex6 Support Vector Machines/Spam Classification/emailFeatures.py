#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def email_features(word_indices):
    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros((n, 1))
    for i in range(word_indices.size):
        x[word_indices[i]] = 1
    return x
