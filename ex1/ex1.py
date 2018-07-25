import numpy as np


def pause_func():
    while input() != '':
        pass


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data


def warm_up_exercise():
    print(np.eye(5))


if __name__ == '__main__':
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')
    warm_up_exercise()
    print('Program paused. Press enter to continue.\n')
    pause_func()
    data1 = load_data("ex1data1.txt")
    print(data1)
