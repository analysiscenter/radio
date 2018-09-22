""" Auxiliary jit-decorated functions for splitting/assembling arrays into/from patches """

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def get_patches_numba(images, shape, stride, out):
    """ Get all patches from array of padded 3D scans, put them into out.

    Parameters
    ----------
    images : ndarray
        4darray, array of 3d-scans.
        assumes scans are already padded.
    shape : ndarray
        3D array with shape of patch (z,y,x).
    stride : ndarray
        3D array with strides of a patch along (z,y,x).
        (if not equal to patch_shape, patches will overlap).
    out : ndarray
        resulting 5d-array, where all patches are put. The first dimension enumerates scans,
        while the second one enumerates patches.
    """

    # for convenience put scan-shape in ndarray
    image_shape = np.zeros(3)
    image_shape[:] = images.shape[1:]

    # compute number of patches along all axes
    num_sections = (image_shape - shape) // stride + 1

    # iterate over patches, put them into out
    for it in prange(images.shape[0]):                                     # pylint: disable=not-an-iterable
        ctr = 0
        for ix in range(int(num_sections[0])):
            for iy in range(int(num_sections[1])):
                for iz in range(int(num_sections[2])):
                    inds = np.array([ix, iy, iz])
                    lx, ly, lz = inds * stride
                    ux, uy, uz = inds * stride + shape
                    out[it, ctr, :, :, :] = images[it, lx:ux, ly:uy, lz:uz]
                    ctr += 1

@njit(parallel=True)
def assemble_patches(patches, stride, out):
    """ Assemble patches into a set of 3d ct-scans with shape scan_shape,
    put the scans into out.

    Parameters
    ----------
    patches : ndarray
        5d array of patches. First dim enumerates scans, while the second
        enumerates patches; other dims are spatial with order (z,y,x).
    stride : ndarray
        stride to extract patches in (z,y,x) dims.
    out : ndarray
        4d-array, where assembled scans are put. First dim enumerates
        scans. Should be filled with zeroes before calling function.

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
    scan_shape = np.zeros(3)
    scan_shape[:] = out.shape[1:]

    # cast patch.shape to ndarray
    patch_shape = np.zeros(3, dtype=np.int64)
    patch_shape[:] = patches.shape[2:]

    # compute the number of sections
    num_sections = (scan_shape - patch_shape) // stride + 1

    # iterate over scans and patches, put them into corresponding place in out
    # increment pixel weight if it belongs to a patch
    weights_inv = np.zeros_like(out)
    for it in prange(out.shape[0]):                                     # pylint: disable=not-an-iterable
        ctr = 0
        for ix in range(int(num_sections[0])):
            for iy in range(int(num_sections[1])):
                for iz in range(int(num_sections[2])):
                    inds = np.array([ix, iy, iz])
                    lx, ly, lz = inds * stride
                    ux, uy, uz = inds * stride + patch_shape
                    out[it, lx:ux, ly:uy, lz:uz] += patches[it, ctr, :, :, :]
                    weights_inv[it, lx:ux, ly:uy, lz:uz] += 1.0
                    ctr += 1

        # weight assembled image
        out[it, ...] /= weights_inv[it]

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
