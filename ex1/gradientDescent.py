import numpy as np
from computeCost import compute_cost


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    j_history = []
    while iterations:
        temp_a = np.dot(x, theta) - y
        theta = theta - (alpha / m) * np.dot(x.T, temp_a)
        j_history.append(compute_cost(x, y, theta))
        iterations -= 1
