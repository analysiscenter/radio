""" Auxiliary jit-decorated functions for splitting/assembling arrays into/from patches """

import numpy as np
from numba import guvectorize, int64, float64


@guvectorize([(float64[:, :, :], int64[:], int64[:], float64[:, :, :, :], int64[:])],
             '(n, m, k),(r),(r),(p, l, s, t)->()',
             nopython=True, target='parallel')
def get_patches_numba(image, shape, stride, out, fake):
    """ Get all patches from padded 3d scan, put them into out.

    Parameters
    ----------
    image : ndarray
        input 3d scan (ct-scan for one patient),
        assumes scan is already padded.
    shape : ndarray
        3D array with shape of patch (z,y,x)
    stride : ndarray
        3D array with strides of a patch along (z,y,x)
        (if not equal to patch_shape, patches will overlap)
    out : ndarray
        resulting 4d-array, where all patches are put. New dimension (first)
        enumerates patches (number_of_patches,z,y,x)
    fake : ndarray
        instrumental array for syntax binding of guvectorize

    """

    # for convenience put image.shape in ndarray
    image_shape = np.zeros(3)
    image_shape[:] = image.shape[:]

    # compute number of patches along all axes
    num_sections = (image_shape - shape) // stride + 1

    # iterate over patches, put them into out
    ctr = 0
    for ix in range(int(num_sections[0])):
        for iy in range(int(num_sections[1])):
            for iz in range(int(num_sections[2])):
                slc_x = slice(ix * stride[0], ix * stride[0] + shape[0])
                slc_y = slice(iy * stride[1], iy * stride[1] + shape[1])
                slc_z = slice(iz * stride[2], iz * stride[2] + shape[2])
                out[ctr, :, :, :] = image[slc_x, slc_y, slc_z]
                ctr += 1


@guvectorize([(float64[:, :, :, :], int64[:], float64[:, :, :], int64[:])],
             '(p, l, s, t),(q),(m, n, k)->()',
             nopython=True, target='parallel')
def assemble_patches(patches, stride, out, fake):
    """ Assemble patches into one 3d ct-scan with shape scan_shape,
    put new scan into out

    Parameters
    ----------
    patches : ndarray
              4d array of patches, first dim enumerates
              patches; other dims are spatial with order (z,y,x)
    stride : ndarray
        stride to extract patches in (z,y,x) dims
    out : ndarray
        3d-array, where assembled scan is put.
        Should be filled with zeroes before calling function.
    fake : ndarray
        instrumental array for syntax binding of guvectorize

    Notes
    -----
    `out.shape`, `stride`, `patches.shape` are used to infer
    the number of sections for each dimension.
    We assume that the number of patches = len(patches)
    corresponds to num_sections.
    Overlapping patches are allowed (stride != patch.shape).
    In this case pixel values are averaged across overlapping patches
    We assume that integer number of patches can be put into
    out using stride.

    """
    out_shape = np.zeros(3)
    out_shape[:] = out.shape[:]

    # cast patch.shape to ndarray
    patch_shape = np.zeros(3)
    patch_shape[:] = patches.shape[1:]

    # compute the number of sections
    num_sections = (out_shape - patch_shape) // stride + 1

    # iterate over patches, put them into corresponding place in out
    # also increment pixel weight if it belongs to a patch
    weights_inv = np.zeros_like(out)
    ctr = 0
    for ix in range(int(num_sections[0])):
        for iy in range(int(num_sections[1])):
            for iz in range(int(num_sections[2])):
                slc_x = slice(ix * stride[0], ix * stride[0] + patch_shape[0])
                slc_y = slice(iy * stride[1], iy * stride[1] + patch_shape[1])
                slc_z = slice(iz * stride[2], iz * stride[2] + patch_shape[2])
                out[slc_x, slc_y, slc_z] += patches[ctr, :, :, :]
                weights_inv[slc_x, slc_y, slc_z] += 1.0
                ctr += 1

    # weight resulting image
    out /= weights_inv


def calc_padding_size(img_shape, patch_shape, stride):
    """ Calculate padding width to add to 3d-scan
        for fitting integer number of patches.

    Parameters
    ----------
    img_shape : ndarray
        shape of 3d-scan along (z,y,x)
    patch_shape : ndarray
        shape of patch along (z,y,x)
    stride : ndarray
        stride to slides over scan, in (z,y,x) dims.

    Returns
    -------
    list or None
        list of tuples with padding sizes
        Pad widths in four dims; the first dim enumerates patients,
        others are spatial axes (z,y,x)
        if no padding is needed, return None
    """
    overshoot = (img_shape - patch_shape + stride) % stride

    pad_delta = np.zeros(3)
    for i in range(len(pad_delta)):                                        # pylint: disable=consider-using-enumerate
        pad_delta[i] = 0 if overshoot[i] == 0 else stride[i] - overshoot[i]

    # calculate and return the padding if not zero
    if np.any(pad_delta > 0):
        before_pad = (pad_delta // 2).astype('int')
        after_pad = (pad_delta - before_pad).astype('int')
        pad_width = [(0, 0)] + [(x, y) for x, y in zip(before_pad, after_pad)]
        return pad_width
    else:
        return None
