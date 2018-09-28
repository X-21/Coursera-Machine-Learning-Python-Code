#!/usr/bin/env python
# -*- coding=utf-8 -*-

from sigmoid import sigmoid


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
