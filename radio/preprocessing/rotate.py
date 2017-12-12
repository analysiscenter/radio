""" Module with jit-compilated functions for rotation operation of 3D scans. """

from numba import jit
import scipy.ndimage


@jit(nogil=True)
def rotate_3D(image, angle, axes=(1, 2)):
    """ Rotate 3D image in plane specified by two axes.

    Parameters
    ----------
    image : ndarray
            3D scan, (z,y,x).
    angle : float
            angle of rotation.
    axes :  tuple
            (int, int), axes that specify rotation plane.

    Returns
    -------
    ndarray
           3D rotated scan

    Notes
    -----
    Zero-padding automatically added after rotation.
    """
    rotated_image = scipy.ndimage.interpolation.rotate(image, angle, axes, reshape=False)
    image[...] = rotated_image[...]
    return None
