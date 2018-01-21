""" Contains auxiliary functions for calculating crop parameters. """

import numpy as np


def make_central_crop(image, crop_size):
    """ Make a crop from center of 3D image of given crop_size.

    Parameters
    ----------
    image : ndarray
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
