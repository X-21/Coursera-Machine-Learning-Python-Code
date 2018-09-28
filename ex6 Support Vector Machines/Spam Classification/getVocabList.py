#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def get_vocab_list():
    return np.loadtxt('../data/vocab.txt', dtype=str)
