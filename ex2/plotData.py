#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# import pandas as pd
# from pandas import DataFrame


def plot_data(x, y):
    # pd_data = DataFrame(data, columns=['Exam 1 score', 'Exam 2 score', 'y'])
    # pd_data_y_plus_1 = pd_data.loc[pd_data['y'] == 1.0][['Exam 1 score', 'Exam 2 score']]
    # pd_data_y_plus_0 = pd_data.loc[pd_data['y'] == 0.0][['Exam 1 score', 'Exam 2 score']]
    # np_data_y_plus_1 = np.array(pd_data_y_plus_1)
    # np_data_y_plus_0 = np.array(pd_data_y_plus_0)

    pos = np.where(y[:, 0] == 1.0)
    neg = np.where(y[:, 0] == 0.0)

    plt.ion()
    plt.figure()
    plt.scatter(x[pos, 0], x[pos, 1], marker="+")
    plt.scatter(x[neg, 0], x[neg, 1], marker="o")
    # plt.xlabel('Exam 1 score')
    # plt.ylabel('Exam 2 score')
    # plt.legend()
    # plt.pause(0.5)
    # plt.close()
