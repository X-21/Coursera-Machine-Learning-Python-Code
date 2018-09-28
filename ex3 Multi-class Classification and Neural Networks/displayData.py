#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def display_data(x, example_width=None):
    m, n = x.shape
    # Set example_width automatically if not passed in
    if not example_width:
        example_width = int(np.round(np.sqrt(n)))
    example_height = int((n / example_width))

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1
    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Get the max value of the patch
            max_val = np.max(np.abs(x[curr_ex, :]))

            wait_set_temp = np.reshape(x[curr_ex, :],
                                       (example_height, example_width), order='F') / max_val
            height_min_temp = pad + (j - 0) * (example_height + pad)
            height_max_temp = height_min_temp + example_height
            width_min_temp = pad + (i - 0) * (example_width + pad)
            width_max_temp = width_min_temp + example_width
            display_array[height_min_temp:height_max_temp, width_min_temp:width_max_temp] = wait_set_temp
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break
    plt.ion()
    plt.imshow(display_array, cmap="gray")  # 选一个漂亮的颜色
    plt.pause(1)
