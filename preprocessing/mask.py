""" helper functions for masks-creation """

import numpy as np


def make_mask(center, diam, x_size, y_size, z_world
              spacing, origin):
    """
    place nodule and
    create slice(2d) of mask given
        x_size, y_size: needed shape of mask
            note that self._data in batch
            is ordered as (z, y, x)
            that is, these args should be like
            (x_size, y_size) = (batch._data.shape[2],
                                batch._data.shape[1])
        spacing: distance between pixels in world coords
                 ordering=(x, y, z)
        origin: world coords of origin on slice
                the nodule needs to be placed on
                ordering=(x, y, z)
        center: world coords of center of nodule
                ordering=(x, y, z)
        z_world: world z-coord (different-slices-direction) of slice
                 the nodule needs to be placed on
    """
    mask = np.zeros([y_size, x_size])

    # pix prefix means 'in pixel coords'
    # center, origin, spacing have xyz-order

    pix_center = (center - origin) / spacing

    # roughly estimate nodule radiuses from above
    # in pixel coords the nodule can be elliptic
    # so, three radiuses
    pix_rad = int(diam / 2 * np.ones_like(center) / spacing + 5)

    # outline window in which nodule-points can be located
    pix_xmin = np.max([0, int(pix_center[0] - pix_rad[0]) - 5])
    pix_xmax = np.min([x_size - 1, int(pix_center[0] + pix_rad[0]) + 5])

    pix_ymin = np.max([0, int(pix_center[1] - pix_rad[1]) - 5])
    pix_ymax = np.min([y_size - 1, int(pix_center[1] + pix_rad[1]) + 5])

    pix_xrange = range(pix_xmin, pix_xmax + 1)
    pix_yrange = range(pix_ymin, pix_ymax + 1)

    # cycle along pixels of window
    for pix_x in pix_xrange:
        for pix_y in pix_yrange:
            # compute world coords of pixel
            world_x = spacing[0] * pix_x + origin[0]
            world_y = spacing[1] * pix_y + origin[1]

            # mark pixel as 1 if it radius-close to nodule center
            # in world coords
            if np.linalg.norm(
                    center - np.array([world_x, world_y, z_world])) <= diam / 2:
                mask[pix_y, pix_x] = 1.0

    return mask
