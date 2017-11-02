"""
Module with auxillary
    jit-compiled functions
    for resize of
    CT scans
"""

from numba import jit
import scipy.ndimage
from PIL import Image
import numpy as np


@jit(nogil=True)
def resize_scipy(patient, out_patient, res, order=3, res_factor=None, padding='edge'):
    """ Resize 3d scan and put it into out_patient.

    Resize engine is scipy.ndimage.interpolation.zoom.
    If res_factor is not supplied, infer resize factor from out_patient.shape.
    otherwise, use res_factor for resize and then crop/pad resized array to out_patient.shape.

    Parameters
    ----------
    patient :     ndarray
                  3D array
    out_patient : ndarray
                  resulting array
    res :         ndarray
                  resulting `skyscraper` for the whole batch.
                  used later by `_post`-func in _inbatch_parallel
    order :       int
                  order of interpolation
    res_factor :  tuple or None
                  resize factor along (z,y,x) in int ir float for interpolation.
                  If not None, can yield array of shape != out_patient.shape,
                  then crop/pad is used
    padding :     str
                  mode of padding, any mode of np.pad()

    Returns
    -------
    tuple
          (res, out_patient.shape), resulting `skyscraper` and shape of
          resized scan inside this `scyscraper`.

    Note
    ----
    Shape of resulting array has to be inferred
    from out_patient
    """
    # infer shape of resulting array
    shape = out_patient.shape

    # define resize factor, perform resizing and put the result into out_patient
    if res_factor is None:
        res_factor = np.array(out_patient.shape) / np.array(patient.shape)
        out_patient[:, :, :] = scipy.ndimage.interpolation.zoom(patient, res_factor,
                                                                order=order)
    else:
        out_patient[:, :, :] = to_shape((scipy.ndimage.interpolation.
                                         zoom(patient, res_factor, order=order)),
                                        shape=shape, padding=padding)

    # return out-array for the whole batch
    # and shape of out_patient
    return res, out_patient.shape


@jit(nogil=True)
def resize_pil(input_array, output_array, res, axes_pairs=None, shape_resize=None,
               resample=None, padding='edge'):
    """ Resize 3D scan.

    Uses _seq_resize over a pair of axes for applying many 2d-resizes,
    then averages over different pairs for obtaining more precise results.

    Parameters
    ----------
    input_array :   ndarray
                    array to be resized.
    ouput_array :   ndarray
                    array, where the result should be put.
    res :           ndarray
                    resulting `skyscraper` for the whole batch.
                    used later by `_post`-func in _inbatch_parallel
    axes_pairs :    tuple, list of tuples or None
                    pairs of axes for 2d resizes, then averaging is performed,
                    e.g., ((0,1),(1,2),(0,2))
                    if None, defaults to ((0, 1), (1, 2))
    shape_resize :  tuple, list, ndarray or None
                    shape of array after resize.
                    If None, infer shape from `ouput_array.shape`.
    resample :      str or None
                    type of PIL resize's resampling method, e.g.
                    `BILINEAR`, `BICUBIC`,`LANCZOS` or `NEAREST`.
                    If None, `BILINEAR` is used.
    padding :       str
                    mode of padding, any mode of np.pad()

    Returns
    -------
    tuple
          (res, out_patient.shape), resulting `skyscraper` and shape of
          resized scan inside this `scyscraper`.
    """
    # if resample not given, set to bilinear
    resample = Image.BILINEAR if resample is None else resample

    # if axes_pairs not supplied, set the arg to two default axes pairs
    axes_pairs = ((0, 1), (1, 2)) if axes_pairs is None else axes_pairs

    # if shape is not supplied, infer it from output_array
    shape_resize = shape_resize if shape_resize is not None else output_array.shape

    if tuple(shape_resize) == output_array.shape:
        for axes in axes_pairs:
            output_array[:, :, :] += _seq_resize(input_array, shape_resize, axes, resample)
    else:
        for axes in axes_pairs:
            output_array[:, :, :] += to_shape(_seq_resize(input_array, shape_resize, axes, resample),
                                              shape=output_array.shape, padding=padding)

    # normalize result of resize (average over resizes with different pairs of axes)
    output_array[:, :, :] /= len(axes_pairs)

    # for post-function
    return res, output_array.shape


@jit(nogil=True)
def _seq_resize(input_array, shape, axes, resample):
    """ Perform 3d-resize based on sequence of 2d-resizes performed on slices.

    Parameters
    ----------
    input_array: ndarray
                 3D array
    shape :      tuple, list or ndarray
                 shape of 3d scan after resize, (z,y,x).
    axes :       tuple, list or ndarray
                 axes for slicing. E.g., `shape` = (z, y, x) and axes = (0, 1). We first loop over
                 2d-slices [i, :, :] and reshape input to shape = (input_array.shape[0], y, x).
                 then loop over slices [:, i, :] and reshape the result to shape = (z, y, x).
    resample :   str or None
                 type of PIL resize's resampling method, e.g.
                 `BILINEAR`, `BICUBIC`,`LANCZOS` or `NEAREST`.
                 If None, `BILINEAR` is used.

    Returns
    -------
    ndarray
            resized 3D array
    """
    result = input_array

    # loop over axes
    for axis in axes:
        slice_shape = np.delete(shape, axis)
        result = _slice_and_resize(result, axis, slice_shape, resample)

    return result


@jit(nogil=True)
def _slice_and_resize(input_array, axis, slice_shape, resample):
    """ Slice 3D array along `axis` and resize each slice to `slice_shape`.

    Parameters
    ----------
    input_array : ndarray
                  3D array
    axis :        int
                  axis along which slices are taken
    slice_shape : tuple,list or ndarray
                  (y,x) shape of each slice after resize
    resample :    str or None
                  type of PIL resize's resampling method, e.g.
                  `BILINEAR`, `BICUBIC`,`LANCZOS` or `NEAREST`.
                  If None, `BILINEAR` is used.

    Returns
    -------
    ndarray
            3D array in which each slice along chosen axis is resized
    """
    # init the resulting array
    result_shape = np.insert(np.array(slice_shape), axis, input_array.shape[axis])
    result = np.zeros(shape=result_shape)

    # invert slice shape for PIL.resize
    slice_shape = slice_shape[::-1]

    # loop over the axis given by axis
    for i in range(result.shape[axis]):
        slices = np.array([slice(None), slice(None), slice(None)])
        slices[axis] = i
        slices = tuple(slices)

        # resize the slice and put the result in result-array
        result[slices] = np.array(Image.fromarray(input_array[slices]).resize(slice_shape, resample=resample))

    return result


def to_shape(data, shape, padding):
    """ Crop or pad 3D array to resize it to `shape`

    Parameters
    ----------
    data :    ndarray
              3D array for reshaping
    shape :   tuple, list or ndarray
              data shape after crop or pad
    padding : str
              mode of padding, any of the modes of np.pad()

    Returns
    -------
    ndarray
            cropped and padded data
    """
    # calculate shapes discrepancy
    data_shape = np.asarray(data.shape)
    shape = np.asarray(shape)
    overshoot = data_shape - shape

    # calclate crop params and perform crop
    crop_dims = np.maximum(overshoot, 0)
    crop_first = crop_dims // 2
    crop_trailing = crop_dims - crop_first
    slices = [slice(first, dim_shape - trailing)
              for first, trailing, dim_shape in zip(crop_first, crop_trailing, data_shape)]
    data = data[slices]

    # calculate padding params and perform padding
    pad_dims = -np.minimum(overshoot, 0)
    pad_first = pad_dims // 2
    pad_trailing = pad_dims - pad_first
    pad_params = [(first, trailing)
                  for first, trailing in zip(pad_first, pad_trailing)]
    data = np.pad(data, pad_width=pad_params, mode=padding)

    # return cropped/padded array
    return data
