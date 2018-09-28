#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y):
    pos = np.where(y[:, 0] == 1)[0]
    neg = np.where(y[:, 0] == 0)[0]
    plt.scatter(x[pos, 0], x[pos, 1], marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], marker='o')
