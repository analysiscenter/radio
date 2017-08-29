""" Module with jit-compilated functions for rotation operation of 3D scans. """

import numpy as np
from numba import jit
import scipy.ndimage


@jit(nogil=True)
def rotate_3D(image, degree, axes=(1, 2)):
    """ Rotate 3D image in plane specified by two axes.

    Args:
    - image: ndarray(l, k, m), 3D image;
    - degree: float, degree of rotation;
    - axes: tuple(int, int), axes that specify rotation plane;

    Returns:
    - ndarray(l, k, m), 3D rotated image;

    *NOTE: zero padding automatically added after rotation;
    """
    return scipy.ndimage.interpolation.rotate(image, degree, axes, reshape=False)


@jit(nogil=True)
def random_rotate_3D(image, max_degree, axes=(1, 2)):
    """ Perform random rotation of input image in plane specified by two axes.

    Args:
    - image: ndarray(l, k, m), 3D image;
    - max_degree: float, max_degree that can be reached by random rotation;
    - axes: tuple(int, int), axes that specify rotation plane;

    Returns:
    - ndarray(l, k, m), 3D rotated image;

    *NOTE: zero padding automatically added after rotation;
    """
    angle = np.random.rand(1) * max_degree
    return scipy.ndimage.interpolation.rotate(image, angle[0], axes, reshape=False)
