import numpy as np


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data


if __name__ == '__main__':
    data1 = load_data("ex1data1.txt")
    print(data1)
