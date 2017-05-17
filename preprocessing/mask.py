""" auxiliary functions for mask-creation """

import numpy as np
from numba import njit


@njit(nogil=True)
def insert_cropped(where, what, st_pos):
    """
    where, what: arrays with same ndims=3
    st_pos: ndarray of length=3
        what-array should be put in where-array starting from st_pos
        what-array is cropped if
            st_pos is negative or
            what-array is too large to be put in where-array
            starting from st_pos

    example:
        where = np.zeros(shape=(3, 3, 3), dtype='int')
        what = np.ones(shape=(2, 2, 2), dtype='int')
        st_pos = np.asarray([2, 2, 2])

        # after execution
        insert_cropped(where, what, st_pos)
        # where[2, 2, 2] = 1, other elems = 0
    """
    # define crop boundaries
    st_what = -np.minimum(np.zeros_like(st_pos), st_pos)
    end_what = np.minimum(np.array(where.shape) - st_pos,
                          np.array(what.shape))

    st_where = np.maximum(st_pos, np.zeros_like(st_pos))
    end_where = np.minimum(st_pos + np.array(what.shape),
                           np.array(where.shape))

    # perform insert
    where[st_where[0]: end_where[0],
          st_where[1]: end_where[1],
          st_where[2]: end_where[2]] = what[st_what[0]: end_what[0],
                                            st_what[1]: end_what[1],
                                            st_what[2]: end_what[2]]


@njit(nogil=True)
def make_mask(batch_mask, img_start, img_end, nodules_start, nodules_size):
    """Make mask using information about nodules location and sizes.

    This function takes batch mask array(batch_mask) filled with zeros,
    start and end pixels of coresponding patient's data array in batch_mask,
    and information about nodules location pixels and pixels sizes.
    Pixels that correspond nodules' locations are filled with ones in
    target array batch_mask.
    """
    for i in range(nodules_start.shape[0]):
        nod_size = nodules_size[0, :]

        nodule = np.ones(int(nod_size[0]),
                         int(nod_size[1]),
                         int(nod_size[2]))

        patient_mask = batch_mask[img_start[0]: img_end[0],
                                  img_start[1]: img_end[1],
                                  img_start[2]: img_end[2]]
        insert_cropped(patient_mask, nodule, nodules_start[i, :])


@njit(nogil=True)
def make_mask_patient(pat_mask, start, size):
    """
    make mask for one patient and put it into pat_mask
    args:
        pat_mask: array where the mask should be put
            the order of axes should be z, y, x
        start: coordinates of nodules' start pixels;
        size: pixel sizes of nodules;
    """

    if size.shape[0] > 0:
        for i in range(size.shape[0]):
            nod_start = start[i, :]
            nod_size = size[i, :]

            nodule = np.ones((int(nod_size[0]),
                              int(nod_size[1]),
                              int(nod_size[2])))

            insert_cropped(pat_mask, nodule, nod_start)
