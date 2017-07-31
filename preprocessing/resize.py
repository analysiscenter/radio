"""
Module with auxillary
    jit-compiled functions
    for resize of
    DICOM-scans
"""

from numba import jit
import scipy.ndimage
from PIL import Image
import numpy as np


@jit(nogil=True)
def resize_scipy(patient, out_patient, res, order=3, res_factor=None, padding='edge'):   # pylint: disable=unused-argument
    """ Resize 3d-scan for one patient and put it into out_patient array.
            Resize engine is scipy.ndimage.interpolation.zoom
            If res_factor is supplied, use this arg for interpolation.
            O/w infer resize factor from out_patient.shape and then crop/pad
            resized array to shape=shape.

    Args
        patient: ndarray with patient data
        out_patient: ndarray, in which patient data
            after resize should be put
        res: out array for the whole batch
            not needed here, will be used later by post default
        order: order of interpolation
        res_factor: resize factor that has to used for the interpolation
            when not None, can yield array of shape != out_patient.shape.
            In this case crop/pad is used
        shape: needed here for correct work
        padding: mode of padding, can be any of the modes used by np.pad
    Return:
        tuple (res, out_patient.shape)

    * shape of resized array has to be inferred
        from out_patient
    """
    # resize-shape is inferred from out_patient
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
def resize_pil(input_array, output_array, res, axes_pairs=None, shape_resize=None, padding='edge'):
    """ Resize 3d-scan for an item given by input_array and put the result in output_array.
            ...
    """
    # if axes_pairs not supplied, set the arg to two default axes pairs
    axes_pairs = ((0, 1), (1, 2)) if axes_pairs is None else axes_pairs

    # if shape is not supplied, infer it from output_array
    shape_resize = shape_resize if shape_resize is not None else output_array.shape

    if shape_resize == output_array.shape:
        for axes in axes_pairs:
            output_array[:, :, :] += _seq_resize(input_array, shape_resize, axes)
    else:
        for axes in axes_pairs:
            output_array[:, :, :] += to_shape(_seq_resize(input_array, shape_resize, axes), shape=output_array.shape,
                                              padding=padding)

    # normalize result of resize (average over resizes with different pairs of axes)
    output_array[:, :, :] /= len(axes_pairs)


@jit(nogil=True)
def _seq_resize(input_array, shape, axes):
    """ Calculates 3d-resize based on sequence of 2d-resizes performed on slices

    Args:
        input_array: 3d-array to be resized
        shape: shape of resizing, tuple/list/ndarray of len=3
        axes: tuple/ndarray/list of len=2 that contains axes that are used for slicing.
            E.g., output_array.shape = (sh0, sh1, sh2) and axes = (0, 1). We first loop over
            2d-slices [i, :, :] and reshape the input to shape = (input_array.shape[0], sh1, sh2).
            We then loop over slices [:, i, :] and reshape the result to shape = (sh0, sh1, sh2).
    Return:
        resized 3d-array
    """
    result = input_array

    # loop over axes
    for axis in axes:
        slice_shape = np.delete(shape, axis)
        result = _slice_and_resize(result, axis, slice_shape)

    return result

@jit(nogil=True)
def _slice_and_resize(input_array, axis, slice_shape):
    """ Slice 3d-array along the axis given by axis-arg and resize each slice to
            shape=slice_shape

        Args:
            input_array: 3d-array to be resized
            axis: axis along which slices are taken
            slice_shape: ndarray/tuple/list of len=2, shape of each slice after resize

        Return:
            3d-array in which each slice along chosen axis is resized
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
        result[slices] = np.array(Image.fromarray(input_array[slices]).resize(slice_shape))

    return result


def to_shape(data, shape, padding):
    """ Crop/pad 3d-array of arbitrary shape s.t. it be a 3d-array
        of shape=shape

    Args:
        data: 3d-array for reshaping
        shape: needed shape, tuple/list/ndaray of len = 3
        padding: mode of padding, can be any of the modes used by np.pad
    Return:
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
