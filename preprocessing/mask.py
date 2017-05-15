""" auxiliary functions for mask-creation """

import numpy as np
from numba import njit


@njit(nogil=True)
def insert_cropped(where, what, origin):
    """
    where, what: arrays with same ndims=3
    origin: ndarray of length=3
        what-array should be put in where-array starting from origin
        what-array is cropped if
            origin is negative or
            what-array is too large to be put in where-array
            starting from origin

    example:
        where = np.zeros(shape=(3, 3, 3), dtype='int')
        what = np.ones(shape=(2, 2, 2), dtype='int')
        origin = np.asarray([2, 2, 2])

        # after execution
        insert_cropped(where, what, origin)
        # where[2, 2, 2] = 1, other elems = 0
    """
    # define crop boundaries
    st_what = -np.minimum(np.zeros_like(origin), origin)
    end_what = np.minimum(np.array(where.shape) - origin, np.array(what.shape))
    st_where = np.maximum(origin, np.zeros_like(origin))
    end_where = np.minimum(origin + np.array(what.shape), np.array(where.shape))

    # perform insert
    where[st_where[0]:end_where[0], st_where[1]:end_where[1], st_where[2]:end_where[2]] = \
        what[st_what[0]:end_what[0], st_what[1]:end_what[1], st_what[2]:end_what[2]]


@njit(nogil=True)
def make_patient_mask(patient_mask, spacing, origin, nodules):
    """
    make mask for one patient and put it into pat_mask
    args:
        patient_mask: array where the mask should be put
            the order of axes should be z, y, x
        spacing: array with spacing (world-distance between pixels) of patient
            order of axes is x, y, z
        origin: array with world coords of pixel[0, 0, 0]
            order of axes is x, y, z
        nodules: ndarray with info about location of patient's nodules
            has shape (number_of_nodules, 4)
            each row corresponds to one nodule
            nodules[i]  = [nod_coord_x, nod_coord_y, nod_coord_z, nod_diam]
            coords, diams given in world coords.
    """

    if len(nodules) > 0:
        # nodule centers in pixel coords
        center_pix = np.rint((nodules[:, :3] - origin) / spacing)

        # recalculate diameters in pixel coords
        # note that there are 3 diams in pixel coords for each nodule
        col_diams = nodules[:, 3].reshape(-1, 1)
        nod_diams = np.rint(col_diams / spacing)

        # nodule starting positions in pix coords
        nod_origin = nod_center_pix - np.rint(nod_diams_pix / 2)

        # loop over nodules (rows in ndarray)
        for i in range(len(nodules)):
            # read info about nodule
            # note that we use z, y, x order in data and mask
            origin_nod = nod_origin[i, :][::-1]
            diams_nod = nod_diams[i, :][::-1]
            # nodule mask is a cube now (but could be a circle)
            nodule = np.ones(diams_nod.astype('int'))

            # insert nodule in mask
            insert_cropped(patient_mask, nodule, origin_nod)
