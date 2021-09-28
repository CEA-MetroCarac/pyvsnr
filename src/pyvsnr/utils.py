"""
Utilities functions
"""
import numpy as np


def pad_centered(arr, shape_ref, value=0):
    """ Pad array with value in a centered way """
    assert (len(shape_ref) == len(arr.shape))

    # to deal with shape_ref smaller than initial shape
    shape_ref = tuple(map(max, arr.shape, shape_ref))

    dim = len(shape_ref)
    arr_pad = arr.copy()
    for k in range(dim):
        # gap between shape_ref and shape_max to pad
        gap = shape_ref[k] - arr.shape[k]
        gap2 = gap // 2

        # swap axes to work on axis=0
        arr_pad = np.swapaxes(arr_pad, 0, k)

        width = (gap2, gap - gap2)
        if dim > 1:
            width = (width,) + (dim - 1) * ((0, 0),)
        arr_pad = np.pad(arr_pad, width, constant_values=value)

        # return to original axis
        arr_pad = np.swapaxes(arr_pad, 0, k)

    return arr_pad
