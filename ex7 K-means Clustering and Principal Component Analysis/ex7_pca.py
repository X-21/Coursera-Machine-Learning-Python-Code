#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat
from scipy.misc import imread

from featureNormalize import feature_normalize
from pca import pca
from projectData import project_data
from recoverData import recover_data
from displayData import display_data
from runkMeans import run_k_means
from kMeansInitCentroids import k_means_init_centroids
from drawLine import draw_line


def pause_func():
    while input() != '':
        pass


def load_mat_file(filename):
    return loadmat(filename)


if __name__ == '__main__':
    # ================== Part 1: Load Example Dataset  ===================
    print('Visualizing example dataset for PCA.\n\n')
    data = load_mat_file('./data/ex7data1.mat')
    X = data['X']
    # Visualize the example dataset
    plt.ion()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    # plt.axis([0.5, 6.5, 2, 8])
    plt.axis("square")
    plt.pause(0.8)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 2: Principal Component Analysis ===============
    print('\nRunning PCA on example dataset.\n\n')
    # Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = feature_normalize(X)
    U, S = pca(X_norm)
    # Draw the eigenvectors centered at mean of data.
    # These lines show the directions of maximum variations in the dataset.

    draw_line(mu, (mu + 1.5 * np.dot(S[0], U[:, 0].T)), "-k")
    draw_line(mu, (mu + 1.5 * np.dot(S[1], U[:, 1].T)), "-k")

    plt.pause(0.8)
    print('Top eigenvector: \n')
    print(' U(:,1) = %s \n' % U[:, 0])
    print('\n(you should expect to see -0.707107 -0.707107)\n')

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =================== Part 3: Dimension Reduction ===================
    # You should now implement the projection step to map the data onto the
    # first k eigenvectors. The code will then plot the data in this reduced
    # dimensional space.  This will show you what the data looks like when
    # using only the corresponding eigenvectors to reconstruct it.
    print('\nDimension reduction on example dataset.\n\n')
    # Plot the normalized dataset (returned from pca)
    plt.close()
    plt.figure()
    plt.scatter(X_norm[:, 0], X_norm[:, 1])
    plt.axis("square")
    plt.pause(0.5)

    # Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example: %s\n' % Z[0])
    print('\n(this value should be about 1.481274)\n\n')

    X_rec = recover_data(Z, U, K)
    print('Approximation of the first example: %s \n', X_rec[0])
    print('\n(this value should be about  -1.047419 -1.047419)\n\n')

    # Draw lines connecting the projected points to the original points
    plt.scatter(X_rec[:, 0], X_rec[:, 1], cmap="r")
    plt.pause(0.5)
    for i in range(X_norm.shape[0]):
        draw_line(X_norm[i], X_rec[i], "--k")
        plt.pause(0.1)

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =============== Part 4: Loading and Visualizing Face Data =============
    # We start the exercise by first loading and visualizing the dataset.
    # The following code will load the dataset into your environment

    # Load Face dataset
    face_date = load_mat_file('./data/ex7faces.mat')
    X = face_date['X']
    plt.close()
    plt.figure()
    display_data(X[0: 100, :])
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 5: PCA on Face Data: Eigenfaces  ===================
    # Run PCA and visualize the eigenvectors which are in this case eigenfaces
    # We display the first 36 eigenfaces.
    print('\nRunning PCA on face dataset.\n(this might take a minute or two ...)\n\n')
    # Before running PCA, it is important to first normalize X by subtracting
    # the mean value from each feature
    X_norm, mu, sigma = feature_normalize(X)

    # Run PCA
    U, S = pca(X_norm)
    # Visualize the top 36 eigenvectors found
    plt.close()
    plt.figure()
    display_data(U[:, 0:36].T)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============= Part 6: Dimension Reduction for Faces =================
    # Project images to the eigen space using the top k eigenvectors
    # If you are applying a machine learning algorithm
    print('\nDimension reduction for face dataset.\n\n')

    K = 100
    Z = project_data(X_norm, U, K)

    print('The projected data Z has a size of: ')
    print(Z.shape)

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
    # Project images to the eigen space using the top K eigen vectors and
    # visualize only using those K dimensions
    # Compare to the original input, which is also displayed

    print('\nVisualizing the projected (reduced dimension) faces.\n\n')
    K = 100
    X_rec = recover_data(Z, U, K)
    # Display normalized data
    plt.close()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original faces')
    display_data(X_norm[0:100, :])

    # Display reconstructed data from only k eigenfaces
    plt.subplot(1, 2, 2)
    plt.title('Recovered faces')
    display_data(X_rec[0:100, :])
    plt.close()

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
    A = imread('./data/bird_small.png')
    A = A / 255
    img_size = A.shape
    X = np.reshape(A, (img_size[0] * img_size[1], 3), order='F')
    K = 16
    pixels_iters = 10
    initial_centroids = k_means_init_centroids(X, K)
    centroids_img, idx_img = run_k_means(X, initial_centroids, pixels_iters)
    # Sample 1000 random indexes (since working with all the data is
    # too expensive. If you have a fast computer, you may increase this.
    sel = (np.floor(np.random.rand(1000, 1) * X.shape[0]) + 1).astype(np.int32)

    _tab20_data = (
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),  # aec7e8
        (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
        (1.0, 0.7333333333333333, 0.47058823529411764),  # ffbb78
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353),  # 98df8a
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
        (1.0, 0.596078431372549, 0.5882352941176471),  # ff9896
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),  # c5b0d5
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451),  # c49c94
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),  # f7b6d2
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),  # c7c7c7
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),  # dbdb8d
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),  # 9edae5
    )
    palette = mcolors.ListedColormap(_tab20_data, N=K).colors
    colors = []
    idx = np.ravel(idx_img[np.ravel(sel)])
    for i in range(len(idx)):
        colors.append(palette[idx[i] - 1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=colors, marker="o")
    ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.pause(1)
    plt.close()

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
    # Use PCA to project this cloud to 2D for visualization
    X_norm, mu, sigma = feature_normalize(X)
    # % PCA and project the data to 2D
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2)

    plt.figure()

    x_temp = Z[(np.ravel(sel)), :]

    # Create palette

    palette = mcolors.ListedColormap(_tab20_data, N=K).colors
    colors = []
    for i in range(len(idx)):
        colors.append(palette[idx[i] - 1])
    plt.scatter(x_temp[:, 0], x_temp[:, 1], c=colors)

    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.pause(1)
    print('Program paused. Press enter to continue.\n')
    # pause_func()
