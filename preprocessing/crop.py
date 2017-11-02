""" Contains auxiliary functions for calculating crop parameters. """

import numpy as np
from numba import njit


def make_central_crop(image, crop_size):
    """ Make a crop from center of 3D image of given crop_size.

    Parameters
    ----------
    image :     ndarray
                3D image of shape `(dim1, dim2, dim3)`.
    crop_size : ndarray or tuple
                Size of crop along three dimensions `(int, int, int)`
    Returns
    -------
    ndarray
            3D crop from image.
    """
    crop_size = np.asarray(crop_size)
    crop_halfsize = np.ceil(crop_size / 2).astype(np.int)
    halfsize = np.rint(np.asarray(image.shape) / 2).astype(np.int)
    cropped_img = image[halfsize[0] - crop_halfsize[0]: halfsize[0] + crop_size[0] - crop_halfsize[0],
                        halfsize[1] - crop_halfsize[1]: halfsize[1] + crop_size[1] - crop_halfsize[1],
                        halfsize[2] - crop_halfsize[2]: halfsize[2] + crop_size[2] - crop_halfsize[2]]
    return cropped_img.copy()


@njit(nogil=True)
def detect_black_border(masked_image):
    """ Get number of black (empty) slices from top and bottom of 3d-scan

    Parameters
    ----------
    masked_image : ndarray
                   3D numpy array
    Returns
    -------
    tuple
         (int,int), numbers of empty slices on top and bottom by z-axis
    """
    n = masked_image.shape[0]
    x_l = 0
    x_u = n - 1

    for i in range(n):
        current_size = masked_image[:i, :, :][masked_image[:i, :, :]].size
        if current_size != 0:
            break
        else:
            x_l = i

    for i in range(n - 1, -1, -1):
        current_size = masked_image[i:, :, :][masked_image[i:, :, :]].size
        if current_size != 0:
            break
        else:
            x_u = i

    if x_l >= x_u:
        return 0, 0

    return x_l, x_u


@njit(nogil=True)
def return_black_border_array(input_image, background=-2000):
    """ Get array with a black (empty) border.

    Parameters
    ----------
    input_image : ndarray
                  3D numpy array
    background : int
                 voxel's value for background color (air in case of CT).
                 default is -2000, it is air in HU scale.
    Returns
    -------
    ndarray
            info with top and bottom number of black (empty) slices,
            and mean value for non-empty part of scan.

    """
    out_array = np.zeros((3, 3))

    masked_image = (input_image != background)

    for axis in range(3):
        data = np.moveaxis(masked_image, axis, 0)

        data_for_mean = np.moveaxis(masked_image, axis, -1)

        _centr = (data_for_mean * np.arange(data_for_mean.shape[-1])).sum() / (data_for_mean.sum() + 0.01)

        (l, u) = detect_black_border(data)

        out_array[axis, 0] = l
        out_array[axis, 1] = u
        out_array[axis, 2] = _centr

    return out_array
