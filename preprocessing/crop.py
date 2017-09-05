""" Contains auxiliary functions for calculating crop parameters """
import numpy as np
from numba import njit


@njit(nogil=True)
def make_central_crop(image, crop_size):
    """ Make crop from from center of 3D image;
    This function returns crop from center of the source image(image argument)
    with given size(crop_size argument).

    Args:
    - image: ndarray(l, k, j), source 3D image;
    - crop_size: ndarray(3) or tuple(int, int, int) size of crop along three dimensions;

    Returns:
    - ndarray, 3D crop of source image;
    """
    crop_size = np.asarray(crop_size)
    crop_halfsize = np.ceil(crop_size / 2).astype(np.int)
    halfsize = np.rint(np.asarray(image.shape) / 2).astype(np.int)
    cropped_img = image[halfsize[0] - crop_halfsize[0]: halfsize[0] + crop_size[0] - crop_halfsize[0],
                        halfsize[1] - crop_halfsize[1]: halfsize[1] + crop_size[1] - crop_halfsize[1],
                        halfsize[2] - crop_halfsize[2]: halfsize[2] + crop_size[2] - crop_halfsize[2]]
    return cropped_img


@njit(nogil=True)
def detect_black_border(masked_image):
    """
    returns number of black slices from top and bottom of 3d-scan
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
    """
    return an array that contains black border
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
