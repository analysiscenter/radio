# pylint: disable=invalid-name
# pylint: disable=missing-docstring
""" Numba-rized functions for XIP intensity projection (maximum, minimum, average) calculation """

import math
import numpy as np
from numba import jit


PROJECTIONS = {
    'axial': [0, 1, 2],
    'coronal': [1, 0, 2],
    'sagital': [2, 0, 1]
}


REVERSE_PROJECTIONS = {
    'axial': [0, 1, 2],
    'coronal': [1, 0, 2],
    'sagital': [1, 2, 0]
}


MODES = {
    'max': 0,
    'min': 1,
    'mean': 2,
    'median': 3
}


@jit(nogil=True, nopython=True)
def maximum_filter1d(data, out):
    """ Compute maximum intensity projection along zero-axis.

    Parameters
    ----------
    data : ndarray(l, m, n)
        input 3d array for for computing xip operation.
    out : ndarray(m, n)
        output 2d array used to store results of xip operation.
    """
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            out[i, j] = np.max(data[:, i, j])


@jit(nogil=True, nopython=True)
def minimum_filter1d(data, out):
    """ Compute minimum intensity projection along zero-axis.

    Parameters
    ----------
    data : ndarray(l, m, n)
        input 3d array for for computing xip operation.
    out : ndarray(m, n)
        output 2d array used to store results of xip operation.
    """
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            out[i, j] = np.min(data[:, i, j])


@jit(nogil=True, nopython=True)
def average_filter1d(data, out):
    """ Compute average intensity projection along zero-axis.

    Parameters
    ----------
    data : ndarray(l, m, n)
        input 3d array for for computing xip operation.
    out : ndarray(m, n)
        output 2d array used to store results of xip operation.
    """
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            out[i, j] = np.mean(data[:, i, j])


@jit(nogil=True, nopython=True)
def median_filter1d(data, out):
    """ Compute median intensity projection along zero-axis.

    Parameters
    ----------
    data : ndarray(l, m, n)
        input 3d array for for computing xip operation.
    out : ndarray(m, n)
        output 2d array used to store results of xip operation.
    """
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            sorted_data = np.sort(data[:, i, j])
            out[i, j] = sorted_data[math.ceil(data.shape[0] / 2)]


@jit(nogil=True, nopython=True)
def numba_xip(image, depth, mode, step):
    """ Apply xip operation to scan of one patient.

    Parameters
    ----------
    data : ndarray
        3d array, patient scan or its crop.
    depth : int
        number of neighborhood points along zero axis for xip kernel function.
    mode : int
        one of following values: 0, 1, 2, 3, that correspond to
        'maximum', 'minimum', 'mean' and 'median' projections.
    stride : int
        stride-step along zero axis.

    Returns
    -------
    ndarray(m, k, l)
        image after xip operation transform.
    """
    size = (image.shape[0] - 2 * math.floor(depth / 2)) // step
    out_data = np.empty(shape=(size, image.shape[1], image.shape[2]), dtype=image.dtype)
    for i in range(0, size):
        ix = i * step
        if mode == 0:
            maximum_filter1d(image[ix: ix + depth, ...], out_data[i, ...])
        elif mode == 1:
            minimum_filter1d(image[ix: ix + depth, ...], out_data[i, ...])
        elif mode == 2:
            average_filter1d(image[ix: ix + depth, ...], out_data[i, ...])
        elif mode == 3:
            median_filter1d(image[ix: ix + depth, ...], out_data[i, ...])
    return out_data


def make_xip_numba(image, depth, stride=1, mode='max', projection='axial', padding='reflect'):
    """ Compute intensity projection (maximum, minimum, mean or median) on input 3d image.

    Popular radiological transformation: max, min, mean or median applyied along an axis.
    Notice that axis is chosen according to projection argument.

    Parameters
    ----------
    image : ndarray(k,l,m)
        input 3D image corresponding to CT-scan or its crop
    stride : int
        stride-step along axis, to apply the func.
    depth : int
        depth of slices (aka `kernel`) along axe made on each step for computing.
    mode : str
        Possible values are 'max', 'min', 'mean' or 'median'.
    projection : str
        Possible values: 'axial', 'coronal', 'sagital'.
        In case of 'coronal' and 'sagital' projections tensor
        will be transposed from [z,y,x] to [x,z,y] and [y,z,x].

    Returns
    -------
    ndarray
        resulting ndarray after kernel function is applied.

    """
    padding_lower, padding_upper = math.floor(depth / 2), math.floor(depth / 2)

    extra_padding = (image.shape[0] - padding_lower - padding_upper) % stride

    padding_lower += math.floor(extra_padding / 2)
    padding_upper += math.floor(extra_padding / 2)

    image_tr = image.transpose(PROJECTIONS[projection])
    image_tr = np.pad(image_tr, [(padding_lower, padding_upper),
                                 (0, 0), (0, 0)], mode=padding)

    result = numba_xip(image_tr, step=stride, depth=depth, mode=MODES[mode])
    return result.transpose(REVERSE_PROJECTIONS[projection])
