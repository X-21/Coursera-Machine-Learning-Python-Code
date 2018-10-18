# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def draw_line(p1, p2,plot_kwargs=""):
    # my tips: in plot(x,y) , every element in x will be place in x-axis,
    # so use this function to draw a line.
    k0 = np.array(([p1[0], p2[0]]))
    k1 = np.array(([p1[1], p2[1]]))
    plt.plot(k0, k1,plot_kwargs)
