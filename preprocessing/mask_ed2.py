""" auxiliary functions for mask-creation """

import numpy as np
from numba import njit


@njit(nogil=True)
def insert_cropped(where, what, st_pos):
    """
    where, what: arrays with same ndims=3
    st_pos: tuple/list/array of length = 3
        what-array should be put in where-array starting from st_pos

    example:
        where = np.zeros(shape=(3, 3, 3), dtype='int')
        what = np.ones(shape=(2, 2, 2), dtype='int')
        st_pos = (2, 2, 2)

        # after execution
        insert_cropped(where, what, st_pos)
        # now where[2, 2, 2] = 1, other elems = 0
    """
    req_pos = np.zeros(shape=3)
    for i in range(3):
        req_pos[i] = what.shape[i] + st_pos[i]

    for i_x in range(max(st_pos[0], 0), min(where.shape[0], req_pos[0])):
        for i_y in range(max(st_pos[1], 0), min(where.shape[1], req_pos[1])):
            for i_z in range(max(st_pos[2], 0), min(where.shape[2], req_pos[2])):
                where[i_x, i_y, i_z] = what[i_x - st_pos[0], i_y - st_pos[1],
                                            i_z - st_pos[2]]


@njit(nogil=True)
def make_mask_patient(pat_mask, spacing, origin, nodules):
    """
    make mask for one patient and put it into pat_mask
    args:
        pat_mask: array where the mask should be put
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
        # nodule locs (centers) in pixel coords
        nod_locs_pix = np.rint((nodules[:, :3] - origin) / spacing)

        # recalculate diameters in pixel coords
        # note that there are 3 diams in pixel coords for each nodule
        col_diams = nodules[:, 3]

        nod_diams_pix = np.zeros_like(nod_locs_pix)
        for i in range(3):
            nod_diams_pix[:, i] = col_diams

        nod_diams_pix = np.rint(nod_diams_pix / spacing)

        # nodule starting positions in pix coords
        nod_locs_st_pos = nod_locs_pix - np.rint(nod_diams_pix / 2)

        # loop over nodules (rows in ndarray)
        for i in range(len(nodules)):
            # read info about nodule
            # note that we use z, y, x order in data and mask
            st_pos_nod = nod_locs_st_pos[i, :][::-1]
            nod_size = nod_diams_pix[i, :][::-1]
            nodule = np.ones((int(nod_size[0]), int(nod_size[1]), int(nod_size[2])))

            # insert nodule in mask
            insert_cropped(pat_mask, nodule, st_pos_nod)
