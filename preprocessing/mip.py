import numpy as np
import numba
from numba import njit


@njit
def numbaMax(arr: np.ndarray, l: int, m: int, n: int) -> np.ndarray:
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


@njit
def numbaMin(arr: np.ndarray, l: int, m: int, n: int) -> np.ndarray:
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


@njit
def numbaAvg(arr: np.ndarray, l: int, m: int, n: int) -> np.ndarray:
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


@njit
def jit_MAX_XIP(image: np.ndarray, start: int = 0, stop: int = -1,
                step: int = 2, depth: int = 10) -> np.ndarray:
    """
    This function takes 3d picture represented by np.ndarray image,
    start position for 0-axis index, stop position for 0-axis index,
    step parameter which represents the step across 0-axis and, finally,
    depth parameter which is associated with the depth of slices across
    0-axis made on each step for computing MAX.
    Code inside the body of function is precompiled
    with numba.
    """
    p = image
    if stop == -1:
        stop = p.shape[0] - depth
    elif (stop + depth) > p.shape[0]:
        stop = p.shape[0] - depth

    out_array = np.zeros((int((stop - start) / step + 1),
                          p.shape[1], p.shape[2]), dtype=image.dtype)
    counter = 0
    for x in range(start, stop + 1, step):
        out_array[counter, :, :] = numbaMax(p[x: x + depth], depth,
                                            p.shape[1], p.shape[2])
        counter += 1

    return out_array


@njit
def jit_MIN_XIP(image: np.ndarray, start: int = 0, stop: int = -1,
                step: int = 2, depth: int = 10) -> np.ndarray:
    """
    This function takes 3d picture represented by np.ndarray image,
    start position for 0-axis index, stop position for 0-axis index,
    step parameter which represents the step across 0-axis and, finally,
    depth parameter which is associated with the depth of slices across
    0-axis made on each step for computing MIN.
    Code inside the body of function is precompiled
    with numba.
    """
    p = image
    if stop == -1:
        stop = p.shape[0] - depth
    elif (stop + depth) > p.shape[0]:
        stop = p.shape[0] - depth

    out_array = np.zeros((int((stop - start) / step + 1),
                          p.shape[1], p.shape[2]), dtype=image.dtype)
    counter = 0
    for x in range(start, stop + 1, step):
        out_array[counter, :, :] = numbaMin(p[x: x + depth], depth,
                                            p.shape[1], p.shape[2])
        counter += 1

    return out_array


@njit
def jit_AVG_XIP(image: np.ndarray, start: int = 0, stop: int = -1,
                step: int = 2, depth: int = 10) -> np.ndarray:
    """
    This function takes 3d picture represented by np.ndarray image,
    start position for 0-axis index, stop position for 0-axis index,
    step parameter which represents the step across 0-axis and, finally,
    depth parameter which is associated with the depth of slices across
    0-axis made on each step for computing MEAN.
    Code inside the body of function is precompiled
    with numba.
    """
    p = image
    if stop == -1:
        stop = p.shape[0] - depth
    elif (stop + depth) > p.shape[0]:
        stop = p.shape[0] - depth

    out_array = np.zeros((int((stop - start) / step + 1),
                          p.shape[1], p.shape[2]), dtype=np.float64)
    counter = 0
    for x in range(start, stop + 1, step):
        out_array[counter, :, :] = numbaAvg(p[x: x + depth], depth,
                                            p.shape[1], p.shape[2])
        counter += 1

    return out_array


def image_XIP(image: np.ndarray, start: int = 0, stop=None,
              step: int = 2, depth: int = 10,
              func: str = 'max', projection: str = "axial",
              verbose: bool = False) -> np.ndarray:
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

    projs = {"axial": [0, 1, 2],
             "coronal": [1, 0, 2],
             "sagital": [2, 0, 1]}

    jit_functions_dict = {'max': jit_MAX_XIP,
                          'min': jit_MIN_XIP,
                          'mean': jit_AVG_XIP}

    p = image.transpose(projs[projection])
    if p.shape[0] < depth:
        raise IndexError('Depth is bigger than ' +
                         'corresponing array dimension')

    if stop is None:
        stop = p.shape[0] - depth
        if verbose:
            print("Stop value changed to ", stop)
    elif (stop + depth) > p.shape[0]:
        stop = p.shape[0] - depth
        if verbose:
            print("Stop value changed to ", stop)

    result = jit_functions_dict[func](p, start,
                                      stop, step, depth)
    return result

if __name__ == '__main__':
    pass
