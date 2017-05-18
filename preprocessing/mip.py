""" Numba-rized functions for MIP calculation """

from functools import partial, wraps
import numpy as np
from numba import njit


_PROJECTIONS = {"axial": [0, 1, 2],
                "coronal": [1, 0, 2],
                "sagital": [2, 0, 1]}

_REVERSE_PROJECTIONS = {"axial": [0, 1, 2],
                        "coronal": [1, 0, 2],
                        "sagital": [1, 2, 0]}

_NUMBA_FUNC = {'max': 0, 'min': 1, 'mean': 2}


@njit(nogil=True)
def min_max_sum_fn(a: float, b: float, flag: int) -> float:
    """Apply njit binary opertaion to a, b.

    Binary operation defined by flag of type int.
    Args:
    - a: int or float, left operand;
    - b: int or float, right operand;
    - flag: int, one of [0, 1, 2] values.
            0 value corresponds to max function;
            1 value corresponds to min function;
            2 value corresponds to sum function;
    """
    if flag == 0:
        return max(a, b)
    elif flag == 1:
        return min(a, b)
    elif flag == 2:
        return a + b
    return 0

@njit(nogil=True)
def numba_xip(arr, l, m, n, flag, fill_value):
    """Compute njit xip for given slice.

    Args:
    - arr: ndarray(l, m, n) with source slice's data;
    - l: int;
    - m: int;
    - n: int;
    - flag: int, each of [0, 1, 2] corresponds to max, min and average
    functions for xip operation;
    """
    res = np.full((m, n), fill_value, dtype=arr.dtype)
    for j in range(m):
        for k in range(n):
            for i in range(l):
                res[j, k] = min_max_sum_fn(res[j, k], arr[i, j, k], flag)
    return res


@njit(nogil=True)
def make_xip(data: np.ndarray, step: int, depth: int,
             start: int = 0, stop: int = -1, func: int = 0, fill_value: int = 0):
    """Apply xip operation to CTImage scan of one patient.

    This function takes 3d picture represented by np.ndarray image,
    start position for 0-axis index, stop position for 0-axis index,
    step parameter which represents the step across 0-axis and, finally,
    depth parameter which is associated with the depth of slices across
    0-axis made on each step for computing MAX.
    Code inside the body of function is precompiled
    with numba.
    """
    if data.shape[0] < depth:
        depth = data.shape[0]

    if stop < 0 or stop + depth > data.shape[0]:
        stop = data.shape[0] - depth

    new_shape = [0, data.shape[1], data.shape[2]]
    new_shape[0] = (stop - start) // step + 1
    out_array = np.zeros((new_shape[0], new_shape[1], new_shape[2]), dtype=data.dtype)
    counter = 0
    for x in range(start, stop + 1, step):
        out_array[counter, :, :] = numba_xip(data[x: x + depth],
                                             depth,
                                             data.shape[1],
                                             data.shape[2],
                                             func, fill_value)
        counter += 1

    return out_array


def xip_fn_numba(func='max', projection="axial", step=2, depth=10):
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
    _projection = _PROJECTIONS[projection]
    _reverse_projection = _REVERSE_PROJECTIONS[projection]
    _function = _NUMBA_FUNC[func]
    def out_function(data: np.ndarray, start: int = 0, end: int = -1, *args, **kwargs):
        data_tr = data.transpose(_projection)
        if _function == 0:
            fill_value =np.finfo(data.dtype).min
        elif _function == 1:
            fill_value = np.finfo(data.dtype).max
        else:
            fill_value = 0
        result = make_xip(data_tr, step=step, depth=depth,
                          start=start, stop=end,
                          func=_function, fill_value=fill_value)
        result.transpose(_reverse_projection)
        return result / depth if _function == 2 else result
    return out_function
