# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def select_threshold(yval, pval):
    yval = np.ravel(yval)
    pval = np.ravel(pval)
    best_epsilon = 0
    best_f1 = 0
    _f1 = 0

    step_size = (pval.max() - pval.min()) / 1000
    for epsilon in np.arange(pval.min(), pval.max(), step_size):
        predictions = (pval < epsilon)

        tp = np.sum(np.logical_and(yval == 1, predictions == 1))
        fp = np.sum(np.logical_and(yval == 0, predictions == 1))
        fn = np.sum(np.logical_and(yval == 1, predictions == 0))

        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            _f1 = (2 * precision * recall) / (precision + recall)
        else:
            _f1 = 0

        if _f1 > best_f1:
            best_f1 = _f1
            best_epsilon = epsilon
    return best_epsilon, best_f1
