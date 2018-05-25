""" Auxiliary functions for mask-creation """
import math
import numpy as np
from numba import njit


@njit(nogil=True)
def create_mask_reg_jit(masks, start, end):
    """ Jit-decorated function for fast computation of masks by regression data.

    This function is usually called inside create_mask_reg function.
    """
    num_items = start.shape[0]
    for i in range(num_items):
        masks[i,
              start[i, 0]: end[i, 0],
              start[i, 1]: end[i, 1],
              start[i, 2]: end[i, 2]] = 1.
    return masks


def create_mask_reg(centers, sizes, probs, crop_shape, threshold):
    """ Create mask by data contained in predictions of regression model. """
    n_items = centers.shape[0]
    masks_array = np.zeros(shape=(n_items, *crop_shape), dtype=np.float)
    _crop_shape = np.asarray(crop_shape)

    start_pixels = np.rint(np.clip(centers - sizes / 2, 0, 1) * _crop_shape).astype(np.int)
    end_pixels = np.rint(np.clip(centers + sizes / 2, 0, 1) * _crop_shape).astype(np.int)
    positions = np.array([p > threshold for p in probs])

    masks_array[positions, ...] = create_mask_reg_jit(masks_array[positions, ...],
                                                      start_pixels[positions, ...],
                                                      end_pixels[positions, ...])
    return masks_array


@njit(nogil=True)
def insert_cropped(where, what, origin):
    """ Insert `what` into `where` starting from `origin`

    Parameters
    ----------
    where : ndarray
        3d-array, in which to insert new data.
    what : ndarray
        3d-array, which is inserted.
    origin : ndarray
        starting positions of insertion along (z,y,x).

    Returns
    -------
    None
        changes `where` array.

    Notes
    -----
    What-array is cropped if origin<0 or what-array
    is too large to be put in where-array.

    Examples
    --------
        where = np.zeros(shape=(3, 3, 3), dtype='int')
        what = np.ones(shape=(2, 2, 2), dtype='int')
        origin = np.asarray([2, 2, 2])

        # after execution
        insert_cropped(where, what, origin)
        # where[2, 2, 2] = 1, other elems = 0
    """
    # shapes to arrays for convenience
    what_shape = np.array(what.shape)
    where_shape = np.array(where.shape)

    # check if there is anything to insert
    if np.any(what_shape + origin <= 0) or np.any(origin >= where_shape):
        return

    # define crop boundaries
    st_what = -np.minimum(np.zeros_like(origin), origin)
    end_what = np.minimum(where_shape - origin, what_shape)

    st_where = np.maximum(origin, np.zeros_like(origin))
    end_where = np.minimum(origin + what_shape, where_shape)

    # perform insert
    where[st_where[0]: end_where[0],
          st_where[1]: end_where[1],
          st_where[2]: end_where[2]] = what[st_what[0]: end_what[0],
                                            st_what[1]: end_what[1],
                                            st_what[2]: end_what[2]]

@njit
def draw_ellipsoid_numba(a, b, c):
    """ Draw ellipsoid with given sizes of axes.

    Parameters
    ----------
    a : int
        size along z axis.
    b : int
        size along y axis.
    c : int
        size along x axis.

    Returns
    -------
    ndarray(2a, 2b, 2c)
    """
    volume = np.zeros(shape=(2 * a, 2 * b, 2 * c))
    for iz in range(-a, a):
        y_lim = math.ceil(b * np.sqrt(1 - (iz / a) ** 2))
        for iy in range(-y_lim, y_lim):
            x_lim = math.ceil(c * np.sqrt(1 - (iz / a) ** 2 - (iy / b) ** 2))
            for ix in range(-x_lim, x_lim):
                if (iz / a) ** 2 + (iy / b) ** 2 + (ix / c) ** 2 <= 1:
                    volume[iz + a, iy + b, ix + c] = 1
    return volume


@njit(nogil=True)
def make_mask_numba(batch_mask, start, end, nodules_start, nodules_size, mode=0):
    """ Make mask using information about nodules location and sizes.

    Takes batch_masks already filled with zeros,
    `img` and `img` positions of coresponding patient's data array in batch_mask,

    Parameters
    ----------
    batch_mask : ndarray
        `masks` from batch, just initialised (filled with zeroes).
    start : ndarray
        for each nodule, start position of patient in `skyscraper` is given
        by (nodule_index, z_start, y_start, x_start)
    end : ndarray
        for each nodule, end position of patient in `skyscraper` is given
        by (nodule_index, z_start, y_start, x_start)
    nodules_start : ndarray(4,)
        array, first dim is nodule index, others (z,y,x)
        are start coordinates of nodules
        (smallest voxel with nodule).
    nodules_size : tuple, list or ndarray
        (z,y,x) shape of nodule

    """
    for i in range(nodules_start.shape[0]):
        nodule_size = nodules_size[i, :]

        if mode == 0:
            nodule = np.ones((int(nodule_size[0]),
                              int(nodule_size[1]),
                              int(nodule_size[2])))
        elif mode == 1:
            nodule = draw_ellipsoid_numba(math.ceil(nodule_size[0] / 2),
                                          math.ceil(nodule_size[1] / 2),
                                          math.ceil(nodule_size[2] / 2))

        patient_mask = batch_mask[start[i, 0]: end[i, 0],
                                  start[i, 1]: end[i, 1],
                                  start[i, 2]: end[i, 2]]
        insert_cropped(patient_mask, nodule, nodules_start[i, :])
