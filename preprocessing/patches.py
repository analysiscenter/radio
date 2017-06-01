import numpy as np
from numba import njit, jit


@njit
def put_patches_numba(data_padded, patch_shape, stride, out_arr):
    """
    get all patches from padded 3d-img
            put them into array out_arr
    args:
            data_padded: input 4d-tensor with ct-scans
                    assume each 3d-scan is already padded
            patch_shape: ndarray of len=3 with
                    needed shape of patch
            stride: ndarray of len=3 with stride
                    of patch-window
                    (*if not equal to patch_shape, patches will overlap)
            out_arr: resulting 4d-array, where all patches are put
                    new dimension (first) enumerates patches
    """

    # put shape of a 3d-scan in ndarray
    img = data_padded[0]
    img_shape = np.zeros(3)
    for i in range(3):
        img_shape[i] = img.shape[i]

    # compute number of patches along all axes
    num_sections = (img_shape - patch_shape) // stride + 1
    ctr = 0

    # iterate over patients
    for ipat in range(data_padded.shape[0]):
        img = data_padded[ipat]
        # iterate over patches, put them into out_arr
        for ix in range(int(num_sections[0])):
            for iy in range(int(num_sections[1])):
                for iz in range(int(num_sections[2])):
                    slc_x = slice(ix * stride[0], ix * stride[0] + patch_shape[0])
                    slc_y = slice(iy * stride[1], iy * stride[1] + patch_shape[1])
                    slc_z = slice(iz * stride[2], iz * stride[2] + patch_shape[2])
                    out_arr[ctr, :, :, :] = img[slc_x, slc_y, slc_z]
                    ctr += 1
