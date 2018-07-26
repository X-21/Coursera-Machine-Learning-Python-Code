import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data


def warm_up_exercise():
    print(np.eye(5))


if __name__ == '__main__':
    # ==================== Part 1: Basic Function ====================
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')
    warm_up_exercise()
    # pause_func()
    # ======================= Part 2: Plotting =======================
    data1 = load_data("ex1data1.txt")
    X = data1[:, 0]
    y = data1[:, 1]
    m = len(y)
    X = X.reshape(m, 1)
    y = y.reshape(m, 1)
    plt.plot(X, y, 'rx')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()
    # pause_func()
    # =================== Part 3: Cost and Gradient descent ===================
    X = np.append(np.ones((m, 1)), X, axis=1)  # Add a column of ones to x
    theta = np.zeros((2, 1))  # initialize fitting parameters
