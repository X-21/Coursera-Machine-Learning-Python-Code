#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from sigmoid import sigmoid


def predict(theta, x):
    return np.floor(sigmoid(np.dot(x, theta)) + 0.5)
