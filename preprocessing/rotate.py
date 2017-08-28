""" Module with jit-compilated functions for rotation operation of 3D scans. """

import numpy as np
from numba import jit
import scipy.ndimage


@jit(nogil=True)
def rotate_3D(image, degree, axes=(1, 2)):
    """ Rotate 3D image along two axes.

    Args:
    - image: ndarray(l, k, m), 3D image;
    - degree: float, degree of rotation;
    - axes: tuple(int, int), axes that specify rotation plane;

    Returns:
    - ndarray(l, k, m), 3D rotated image;

    *NOTE: zero padding automatically added after rotation;
    """
    rotated_image = scipy.ndimage.interpolation.rotate(data, degree, axes)
    return rotated_image
