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
def make_mask_patient(pat_mask, center, size):
    """
    make mask for one patient and put it into pat_mask
    args:
        pat_mask: array where the mask should be put
            the order of axes should be z, y, x
        center: coordinates of nodules' center pixels;
        size: pixel sizes of nodules;
    """

    if len(size.shape[0]) > 0:
        for i in range(nodules_size.shape[0]):
            nod_start = nod_locs_st_pos[i, :]
            nod_size = nod_diams_pix[i, :]
            nodule = np.ones((int(nod_size[0]), int(nod_size[1]), int(nod_size[2])))
            insert_cropped(pat_mask, nodule, nod_start)
