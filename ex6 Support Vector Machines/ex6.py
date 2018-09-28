#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from plotData import plot_data
from sklearn.svm import SVC

from gaussianKernel import gaussian_kernel


def pause_func():
    while input() != '':
        pass


def load_mat_file(filename):
    return loadmat(filename)


if __name__ == '__main__':
    # =============== Part 1: Loading and Visualizing Data ================
    print('Loading and Visualizing Data ...\n')
    # Load from ex6data1:
    # You will have X, y in your environment
    data = load_mat_file('./data/ex6data1.mat')
    X = data['X']
    y = data['y']
    plt.ion()
    plt.figure()
    plot_data(X, y)
    plt.pause(1)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ==================== Part 2: Training Linear SVM ====================
    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    plt.figure()
    plot_data(X, y)

    C = 1
    Classification = SVC(C=C, kernel='linear')
    # fit(X, y, sample_weight=None), y : array-like, shape (n_samples,)
    Classification.fit(X, y.ravel())

    plot_pad = 0.5
    plot_x_min, plot_x_max = X[:, 0].min() - plot_pad, X[:, 0].max() + plot_pad
    plot_y_min, plot_y_max = X[:, 1].min() - plot_pad, X[:, 1].max() + plot_pad

    plot_step = 0.01
    plot_x, plot_y = np.meshgrid(np.arange(plot_x_min, plot_x_max, plot_step),
                                 np.arange(plot_y_min, plot_y_max, plot_step))
    plot_z = Classification.predict(np.c_[plot_x.ravel(), plot_y.ravel()]).reshape(plot_x.shape)
    plt.contourf(plot_x, plot_y, plot_z, cmap="Wistia", alpha=0.2)

    plt.pause(1)
    plt.close()

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 3: Implementing Gaussian Kernel ===============
    print('\nEvaluating the Gaussian Kernel ...\n')
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)
    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :'
          '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 4: Visualizing Dataset 2 ================
    print('Loading and Visualizing Data ...\n')
    data = load_mat_file('./data/ex6data2.mat')
    X = data['X']
    y = data['y']
    plt.figure()
    plot_data(X, y)
    plt.pause(1)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    plt.figure()
    plot_data(X, y)

    Classification = SVC(C=100, kernel='rbf', gamma=6)
    # fit(X, y, sample_weight=None), y : array-like, shape (n_samples,)
    Classification.fit(X, y.ravel())

    plot_pad = 0.5
    plot_x_min, plot_x_max = X[:, 0].min() - plot_pad, X[:, 0].max() + plot_pad
    plot_y_min, plot_y_max = X[:, 1].min() - plot_pad, X[:, 1].max() + plot_pad

    plot_step = 0.01
    plot_x, plot_y = np.meshgrid(np.arange(plot_x_min, plot_x_max, plot_step),
                                 np.arange(plot_y_min, plot_y_max, plot_step))
    plot_z = Classification.predict(np.c_[plot_x.ravel(), plot_y.ravel()]).reshape(plot_x.shape)
    plt.contourf(plot_x, plot_y, plot_z, cmap="Wistia", alpha=0.2)
    plt.axis([-0.1, 1.1, 0.3, 1.05])
    plt.pause(1)
    plt.close()

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 6: Visualizing Dataset 3 ================
    print('Loading and Visualizing Data ...\n')
    data = load_mat_file('./data/ex6data3.mat')
    X = data['X']
    y = data['y']
    plt.figure()
    plot_data(X, y)
    plt.pause(1)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ========== Part 7: Training SVM with RBF Kernel (Dataset 2) ==========
    plt.figure()
    plot_data(X, y)

    Classification = SVC(C=1, kernel='poly', degree=3, gamma=10)
    # fit(X, y, sample_weight=None), y : array-like, shape (n_samples,)
    Classification.fit(X, y.ravel())

    plot_pad = 0.5
    plot_x_min, plot_x_max = X[:, 0].min() - plot_pad, X[:, 0].max() + plot_pad
    plot_y_min, plot_y_max = X[:, 1].min() - plot_pad, X[:, 1].max() + plot_pad

    plot_step = 0.01
    plot_x, plot_y = np.meshgrid(np.arange(plot_x_min, plot_x_max, plot_step),
                                 np.arange(plot_y_min, plot_y_max, plot_step))
    plot_z = Classification.predict(np.c_[plot_x.ravel(), plot_y.ravel()]).reshape(plot_x.shape)
    plt.contourf(plot_x, plot_y, plot_z, cmap="Wistia", alpha=0.2)
    plt.axis([-0.8, 0.4, -0.8, 0.8])
    plt.pause(1)
    plt.close()

    print('Program paused. Press enter to continue.\n')
    # pause_func()
