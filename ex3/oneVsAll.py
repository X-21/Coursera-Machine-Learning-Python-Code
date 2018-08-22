#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from fminunc_lr import my_fminunc_lr


def one_vs_all(x, y, num_labels, ova_lambda):
    m, n = x.shape
    all_theta = np.zeros((num_labels, n + 1))
    x = np.append(np.ones((m, 1)), x, axis=1)
    for i in range(1, num_labels + 1):
        initial_theta = np.zeros((n + 1, 1))
        y_temp = (y == i).astype(np.int32)
        result = my_fminunc_lr(x, y_temp, initial_theta, ova_lambda)
        print('\nIteration: %4d' % result['nit'], ' | Cost: ', result['fun'])
        all_theta[i - 1, :] = result['x'].T
    return all_theta
