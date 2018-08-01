import numpy as np


def compute_cost(x, y, theta):
    m = len(y)
    j = 0
    h_theta = np.dot(x, theta)
    err = h_theta - y
    err_sum = sum(err ** 2)
    j = err_sum / (2 * m)
    return j
