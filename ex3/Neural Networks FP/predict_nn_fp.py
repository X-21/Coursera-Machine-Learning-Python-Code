#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import sys

sys.path.append('../')
from sigmoid import sigmoid


def predict(theta1, theta2, x):
    m = x.shape[0]
    p = np.zeros((m, 1))
    print('\nSize of x : ', x.shape)
    x = np.append(np.ones((m, 1)), x, axis=1)
    a2 = sigmoid(np.dot(theta1, x.T))
    print('\nSize of a2 : ', a2.shape)
    a2 = np.append(np.ones((1, a2.shape[1])), a2, axis=0)
    a3 = sigmoid(np.dot(theta2, a2))
    print('\nSize of a3 : ', a3.shape)
    for i in range(m):
        p[i] = np.argmax(a3[:, i])
    return p + 1
