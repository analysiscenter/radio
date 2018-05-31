""" Module with jit-compilated functions for rotation operation of 3D scans. """

import math
from numba import njit
import scipy.ndimage


@njit(parallel=True, nopython=True)
def rotate_coords(x, y, center, angle):
    num_points = x.shape[0]
    out_x = np.zeros_like(x)
    out_y = np.zeros_like(y)
    x_, y_ = x - center[0], y - center[1]

    out_x = math.cos(angle) * x_ - math.sin(angle) * y_
    out_y = math.sin(angle) * x_ + math.cos(angle) * y_
    return out_x + center[0], out_y + center[1]


@njit(parallel=True, nopython=True)
def rotate_and_round(x, y, cx, cy, angle):
    out_x = np.cos(angle) * (x - cx) - np.sin(angle) * (y - cy)
    out_y = np.sin(angle) * (x - cx) + np.cos(angle) * (y - cy)
    return int(round(out_x + cx)), int(round(out_y + cy))


@njit(parallel=True, nopython=True)
def rotate_2d(image, angle, center=(0, 0), fill=0):
    sh, sw = image.shape[0], image.shape[1]
    center = np.array(center)
    angle = -angle

    cx, cy = rotate_coords(np.array([0, sw, sw, 0]),
                           np.array([0, 0, sh, sh]),
                           center, angle)

    dw, dh = int(np.ceil(cx.max() - cx.min())), int(np.ceil(cy.max() - cy.min()))
    min_cx, min_cy = cx.min(), cy.min()
    out_image = np.empty(shape=(dh, dw), dtype=image.dtype)
    for idx in range(dw):
        for idy in range(dh):
            isx, isy = rotate_and_round(idx + min_cx, idy + min_cy,
                                        center[0], center[1], -angle)

            if (0 <= isx) and (isx < sw) and (0 <= isy) and (isy < sh):
                out_image[idy, idx] = image[isy, isx]
            else:
                out_image[idy, idx] = fill
    return out_image


@jit(nogil=True)
def rotate_3D(image, angle, axes=(1, 2)):
    """ Rotate 3D image in plane specified by two axes.

    Parameters
    ----------
    image : ndarray
        3D scan, (z,y,x).
    angle : float
        angle of rotation.
    axes :  tuple, list or ndarray
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
    return rotated_image
