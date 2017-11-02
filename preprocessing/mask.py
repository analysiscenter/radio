""" Auxiliary functions for mask-creation """

import numpy as np
from numba import njit


@njit(nogil=True)
def insert_cropped(where, what, origin):
    """ Insert `what` into `where` starting from `origin`

    Parameters
    ----------
    where :  ndarray
             3d-array, in which to insert new data.
    what :   ndarray
             3d-array, which is inserted.
    origin : ndarray
             starting positions of insertion along (z,y,x).

    Returns
    -------
    None
        changes `where` array.

    Note
    ----
    What-array is cropped if origin<0 or what-array
    is too large to be put in where-array.

    Example
    -------
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


@njit(nogil=True)
def make_mask_numba(batch_mask, img_start, img_end, nodules_start, nodules_size):
    """ Make mask using information about nodules location and sizes.

    Takes batch_masks already filled with zeros,
    `img_start` and `img_end` positions of coresponding patient's data array in batch_mask,

    Parameters
    ----------
    batch_mask :    ndarray
                    `masks` from batch, just initialised (filled with zeroes).
    img_start :     ndarray
                    for each nodule, start position of patient in `skyscraper` is given
                    by (nodule_index, z_start, y_start, x_start)
    img_end :       ndarray
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

        nodule = np.ones((int(nodule_size[0]),
                          int(nodule_size[1]),
                          int(nodule_size[2])))

        patient_mask = batch_mask[img_start[i, 0]: img_end[i, 0],
                                  img_start[i, 1]: img_end[i, 1],
                                  img_start[i, 2]: img_end[i, 2]]
        insert_cropped(patient_mask, nodule, nodules_start[i, :])
