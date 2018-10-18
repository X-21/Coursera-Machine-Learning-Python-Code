# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np


def load_movie_list():
    file = open('./data/movie_ids.txt', 'r', encoding='ISO-8859-1')
    n = 1682
    movie_list = []
    ctx = ''
    idx = 0
    while 1:
        try:
            ctx = file.readline()
        except UnicodeDecodeError:
            pass
        if not ctx:
            break
        idx, ctx = ctx.split(' ', 1)
        movie_list.append(ctx)
    movie_list = np.array(movie_list)
    return movie_list
