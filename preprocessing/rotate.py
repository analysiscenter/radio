""" Module with jit-compilated functions for rotation operation of 3D scans. """

import numpy as np
from numba import jit
import scipy.ndimage


@jit(nogil=True)
def rotate_3D(image, angle, axes=(1, 2)):
    """ Rotate 3D image in plane specified by two axes.

    Args:
    - image: ndarray(l, k, m), 3D image;
    - angle: float, angle of rotation;
    - axes: tuple(int, int), axes that specify rotation plane;

    Returns:
    - ndarray(l, k, m), 3D rotated image;

    *NOTE: zero padding automatically added after rotation;
    """
    rotated_image = scipy.ndimage.interpolation.rotate(image, angle, axes, reshape=False)
    image[...] = rotated_image[...]
    return None
