# pylint: disable=invalid-name
# pylint: disable=missing-docstring
""" Numba-rized functions for XIP intensity projection (maximum, minimum, average) calculation """

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
def min_max_sum_fn(a, b, flag):
    """ Apply njit binary opertaion to `a` and `b`.

    Binary operation defined by flag.

    Parameters
    ----------
    a :    int or float
           left operand.
    b :    int or float
           right operand;
    flag : int
           either 0, 1 or 2.
            0 for max function;
            1 for min function;
            2 for sum function.

    Returns
    -------
    int or float
                result of function or 0.
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
    """ Compute njit xip (intensity projection) for given slice.

    Parameters
    ----------
    arr :        ndarray(l, m, n)
                 input array for computing xip.
    l :          int
                 z-dim of `arr`.
    m :          int
                 y-dim of `arr`.
    n :          int
                 x-dim of `arr`.
    flag :       int
                 0, 1 or 2; corresponds to max, min and average
                 functions for xip operation.
    fill_value : int
                 default value to fill resulting array.
    Returns
    -------
    ndarray
            xip ndarray.
    """
    res = np.full((m, n), fill_value, dtype=arr.dtype)
    for j in range(m):
        for k in range(n):
            for i in range(l):
                res[j, k] = min_max_sum_fn(res[j, k], arr[i, j, k], flag)
    return res


@njit(nogil=True)
def make_xip(data, step, depth,
             start=0, stop=-1, func=0, fill_value=0):
    """ Apply xip operation to scan of one patient.

    Parameters
    ----------
    data :       ndarray
                 3d array, patient scan or crop
    step :       int
                 stride-step along axe, to apply the func.
    depth :      int
                 depth of slices (aka `kernel`) along axe made on each step for computing.
    start :      int
                 number of slize by 0-axis to start operation
    end :        int
                 number of slize by 0-axis to stop operation
    func :       int
                 either 0, 1 or 2.
                 0 for max function;
                 1 for min function;
                 2 for sum function.
    fill_value : int
                 default value to fill resulting array.

    Returns
    -------
    ndarray
           xip ndarray
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
    """ Make intensity projection (maximum, minimum, average)

        Popular radiologic transformation: max, min, avg applyied along an axe.
        Notice that axe is chosen in accordance with projection argument.

        Parameters
        ----------
        step :       int
                     stride-step along axe, to apply the func.
        depth :      int
                     depth of slices (aka `kernel`) along axe made on each step for computing.
        func :       str
                     Possible values are 'max', 'min' and 'avg'.
        projection : str
                     Possible values: 'axial', 'coroanal', 'sagital'.
                     In case of 'coronal' and 'sagital' projections tensor
                     will be transposed from [z,y,x] to [x, z, y] and [y, z, x].

        Returns
        -------
        ndarray
               resulting ndarray after func is applied.

        """
    _projection = _PROJECTIONS[projection]
    _reverse_projection = _REVERSE_PROJECTIONS[projection]
    _function = _NUMBA_FUNC[func]

    def out_function(data, start=0, end=-1):
        data_tr = data.transpose(_projection)
        if _function == 0:
            fill_value = np.finfo(data.dtype).min
        elif _function == 1:
            fill_value = np.finfo(data.dtype).max
        else:
            fill_value = 0
        result = make_xip(data_tr, step=step, depth=depth,
                          start=start, stop=end,
                          func=_function, fill_value=fill_value)
        result = result.transpose(_reverse_projection)
        return result / depth if _function == 2 else result
    return out_function
