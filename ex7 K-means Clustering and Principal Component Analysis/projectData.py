#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def project_data(x, u, k):
    k_list = list(range(0, k))
    u_reduce = u[:, k_list]
    return np.dot(x, u_reduce)
