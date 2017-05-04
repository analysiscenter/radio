# pylint: disable=invalid-name
""" Numba-rized functions for MIP calculation """

from functools import partial
import numpy as np
from numba import njit


_projections = {"axial": [0, 1, 2],
                "coronal": [1, 0, 2],
                "sagital": [2, 0, 1]}

_jit_functions = {'max': 0, 'min': 1, 'mean': 2}


@njit(nogil=True)
def numba_max(arr: np.ndarray, l: int, m: int, n: int) -> np.ndarray:
    """
    Takes 3-dimension numpy ndarray view
    with shape parameters as its arguments.
    The function computes maximum
    along 0-axis of the input ndarray view.
    Code inside the body of function is precompiled
    with numba.
    """
    MAX = np.full((m, n), -10000, dtype=arr.dtype)
    for j in range(m):
        for k in range(n):
            for i in range(l):
                if arr[i, j, k] > MAX[j, k]:
                    MAX[j, k] = arr[i, j, k]
    return MAX


@njit(nogil=True)
def numba_min(arr: np.ndarray, l: int, m: int, n: int) -> np.ndarray:
    """
    Takes 3-dimension numpy ndarray view
    with shape parameters as its arguments.
    The function computes minimum
    along 0-axis of the input ndarray view.
    Code inside the body of function is precompiled
    with numba.
    """
    MIN = np.full((m, n), 10000, dtype=arr.dtype)
    for j in range(m):
        for k in range(n):
            for i in range(l):
                if arr[i, j, k] < MIN[j, k]:
                    MIN[j, k] = arr[i, j, k]
    return MIN


@njit(nogil=True)
def numba_avg(arr: np.ndarray, l: int, m: int, n: int) -> np.ndarray:
    """
    Takes 3-dimension numpy ndarray view
    with shape parameters as its arguments.
    The function computes mean value
    along 0-axis of the input ndarray view.
    Code inside the body of function is precompiled
    with numba.
    """
    AVG = np.zeros((m, n), np.float64)
    for j in range(m):
        for k in range(n):
            for i in range(l):
                AVG[j, k] += arr[i, j, k]
            AVG[j, k] /= l
    return AVG


@njit(nogil=True)
def make_xip(func: int, projection: list, step: int, depth: int,
             patient: np.ndarray, start: int=0, stop: int=-1) -> np.ndarray:
    """
    This function takes 3d picture represented by np.ndarray image,
    start position for 0-axis index, stop position for 0-axis index,
    step parameter which represents the step across 0-axis and, finally,
    depth parameter which is associated with the depth of slices across
    0-axis made on each step for computing MAX.
    Code inside the body of function is precompiled
    with numba.
    """
    p = patient.transpose(projection)

    if p.shape[0] < depth:
        depth = p.shape[0]

    if stop < 0 or stop + depth > p.shape[0]:
        stop = p.shape[0] - depth

    new_shape = p.shape
    new_shape[0] = (stop - start) // step + 1
    out_array = np.zeros(new_shape, dtype=patient.dtype)

    if func == 0:
        xip_fn = numba_max
    elif func == 1:
        xip_fn = numba_min
    else:
        xip_fn = numba_avg

    counter = 0
    for x in range(start, stop + 1, step):
        out_array[counter, :, :] = xip_fn(p[x: x + depth], depth, p.shape[1], p.shape[2])
        counter += 1

    return out_array


def numba_xip_fn(func='max', projection="axial", step=2, depth=10):
    """
    This function takes 3d picture represented by np.ndarray image,
    start position for 0-axis index, stop position for 0-axis index,
    step parameter which represents the step across 0-axis and, finally,
    depth parameter which is associated with the depth of slices across
    0-axis made on each step for computing MEAN, MAX, MIN
    depending on func argument.
    Possible values for func are 'max', 'min' and 'avg'.
    Notice that 0-axis in this annotation is defined in accordance with
    projection argument which may take the following values: 'axial',
    'coroanal', 'sagital'.
    Suppose that input 3d-picture has axis associations [z, x, y], then
    axial projection doesn't change the order of axis and 0-axis will
    be correspond to 0-axis of the input array.
    However in case of 'coronal' and 'sagital' projections the source tensor
    axises will be transposed as [x, z, y] and [y, z, x]
    for 'coronal' and 'sagital' projections correspondingly.
    """
    return partial(make_xip, _jit_functions[func], _projections[projection], step, depth)
