# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_my_Set1_data = (
    (0.89411764705882357, 0.10196078431372549, 0.10980392156862745),
    (0.21568627450980393, 0.49411764705882355, 0.72156862745098038),
    (0.30196078431372547, 0.68627450980392157, 0.29019607843137257),
    (0.59607843137254901, 0.30588235294117649, 0.63921568627450975),
    (1.0, 0.49803921568627452, 0.0),
    (1.0, 1.0, 0.2),
    (0.65098039215686276, 0.33725490196078434, 0.15686274509803921),
    (0.96862745098039216, 0.50588235294117645, 0.74901960784313726),
    (0.6, 0.6, 0.6),
)


def plot_data_points(x, idx, k):
    idx = np.ravel(idx)

    # Create palette
    if k > 9:
        print("WARN. function: plot_data_points. colors is't enough\n")
    palette = mcolors.ListedColormap(_my_Set1_data, N=k).colors
    colors = []
    for i in range(len(idx)):
        colors.append(palette[idx[i] - 1])
    plt.scatter(x[:, 0], x[:, 1], c=colors)
