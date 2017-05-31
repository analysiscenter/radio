import numpy as np
from numba import njit, jit


@njit
def get_patches_padded(img, patch_shape, stride, out_arr):
    """
    get all patches from padded 3d-img
            put them into array out_arr
    args:
            img: input 3d-image (ct-scan for one patient)
                    assume img is already padded
            patch_shape: ndarray of len=3 with
                    needed shape of patch
            stride: ndarray of len=3 with stride
                    of patch-window
                    (*if not equal to patch_shape, patches will overlap)
            out_arr: resulting 4d-array, wheere all patches are put
                    new dimension (first) enumerates patches
    """

    # for convenience put img.shape in ndarray
    img_shape = np.zeros(3)
    for i in range(3):
        img_shape[i] = img.shape[i]

    # compute number of patches along all axes
    num_sections = (img_shape - patch_shape) // stride + 1

    # iterate over patches, put them into out_arr
    ctr = 0
    for ix in range(int(num_sections[0])):
        for iy in range(int(num_sections[1])):
            for iz in range(int(num_sections[2])):
                slc_x = slice(ix * stride[0], ix * stride[0] + patch_shape[0])
                slc_y = slice(iy * stride[1], iy * stride[1] + patch_shape[1])
                slc_z = slice(iz * stride[2], iz * stride[2] + patch_shape[2])
                out_arr[ctr, :, :, :] = img[slc_x, slc_y, slc_z]
                ctr += 1
