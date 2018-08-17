#!/usr/bin/env python
# -*- coding=utf-8 -*-

# import numpy as np
from scipy.special import expit


def sigmoid(x):
    # return 1 / (1 + np.exp(-1 * x))
    return expit(x)
