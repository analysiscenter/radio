import numpy as np
from numba import guvectorize, int64, float64


@guvectorize([(float64[:, :, :], int64[:], int64[:], float64[:, :, :, :], int64[:])],
             '(n, m, k),(r),(r),(p, l, s, t)->()',
             nopython=True, target='parallel')
def put_patches_numba(img, patch_shape, stride, out_arr, fake):
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
            out_arr: resulting 4d-array, where all patches are put
                    new dimension (first) enumerates patches
            fake: fake-result array
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



@guvectorize([(float64[:, :, :, :], int64[:], float64[:, :, :], int64[:])],
             '(p, l, s, t),(q),(m, n, k)->()',
             nopython=True, target='parallel')
def assemble_patches(patches, stride, out_arr, fake):
    """
    assemble patches into one 3d ct-scan with shape scan_shape
        put the scan into out_arr
    args:
        patches: 4d-array of patches, first dim enumerates
            patches; other dims are spatial with order (z, y, x)
        stride: ndarray of len=3 with stride with which the patches
            were extracted
        out_arr: 3d-array, where assembled scan is put
            should be filled with zeroes
            *note 1: out_arr.shape, stride, patch shape are used to infer 
                the number of sections for each dimension.
            We assume that the number of patches = len(patches) 
                corresponds to num_sections
            *note 2: overlapping patches are allowed (stride != patch.shape).
                In this case pixel values are averaged across overlapping
                    patches
            *note 3: we assume that integer number of patches can be put into
                out_arr using stride 
        fake: fake-result array
    """
    out_arr_shape = np.zeros(3)
    for i in range(3):
        out_arr_shape[i] = out_arr.shape[i]

    # cast patch.shape to ndarray
    patch_shape = np.zeros(3)
    for i in range(3):
        patch_shape[i] = patches.shape[i + 1]

    # compute the number of sections
    num_sections = (out_arr_shape - patch_shape) // stride + 1

    # iterate over patches, put them into corresponding place in out_arr
    # also increment pixel weight if it belongs to a patch
    weights_inv = np.zeros_like(out_arr)
    ctr = 0
    for ix in range(int(num_sections[0])):
        for iy in range(int(num_sections[1])):
            for iz in range(int(num_sections[2])):
                slc_x = slice(ix * stride[0], ix * stride[0] + patch_shape[0])
                slc_y = slice(iy * stride[1], iy * stride[1] + patch_shape[1])
                slc_z = slice(iz * stride[2], iz * stride[2] + patch_shape[2])
                out_arr[slc_x, slc_y, slc_z] += patches[ctr, :, :, :]
                weights_inv[slc_x, slc_y, slc_z] += 1.0
                ctr += 1

    # weight resulting image
    out_arr /= weights_inv

def calc_padding_size(img_shape, patch_shape, stride):
    """
    calculate width of padding, that needs to be added to 3d-scan 
        in order to fit integer number of patches;
        in format needed for np.pad function
    args:
        img_shape: ndarray of len=3 with shape of 3d-scan
        patch_shape: ndarray of len=3 with shape of a patch
        stride: stride with which patch-window slides over scan
    return: paddding size as list (len=4) of tuples of len=2
        with pad widths in four dims; the first dim enumerates patients,
            others are spatial axes
        if no padding is needed, return None
    """
    overshoot = (img_shape - patch_shape + stride) % stride

    pad_delta = np.zeros(3)
    for i in range(len(pad_delta)):
        pad_delta[i] = 0 if overshoot[i] == 0 else stride[i] - overshoot[i]

    # calculate and return the padding if not zero
    if np.any(pad_delta > 0):
        before_pad = (pad_delta // 2).astype('int')
        after_pad = (pad_delta - before_pad).astype('int')
        pad_width = [(0, 0)] + [(x, y) for x, y in zip(before_pad, after_pad)]
        return pad_width
    else:
        return None

