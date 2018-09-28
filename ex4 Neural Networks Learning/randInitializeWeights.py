#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def rand_initialize_weights(layer_in, layer_out):
    epsilon_init = np.sqrt(6 / (layer_in + layer_out))
    return np.random.rand(layer_out, layer_in + 1) * 2 * epsilon_init - epsilon_init
