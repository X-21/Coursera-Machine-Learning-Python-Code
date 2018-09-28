# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def debug_initialize_weights(fan_out, fan_in):
    # Set W to zeros
    w = np.zeros((fan_out, 1 + fan_in))
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    return np.reshape(np.sin(np.arange(w.size) + 1), w.shape, order='F') / 10
