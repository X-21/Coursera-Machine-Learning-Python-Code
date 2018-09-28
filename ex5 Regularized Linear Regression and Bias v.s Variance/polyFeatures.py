#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def poly_features(x, p):
    m = x.shape[0]
    x_poly = np.zeros((m, p))
    for i in range(p):
        x_poly[:, i] = (x ** (i + 1)).reshape(m, )
    return x_poly
