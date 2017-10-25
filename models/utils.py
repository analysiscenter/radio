import numpy as np
from numba import njit


@njit(nogil=True)
def create_mask_jit(masks, start, end):
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

    masks_array[positions, ...] = create_mask_jit(masks_array[positions, ...],
                                                  start_pixels[positions, ...],
                                                  end_pixels[positions, ...])
    return masks_array


def nodules_info_to_rzyx(nodules, scale=True):
    """ Transform data contained in nodules_info array to rzyx format. """
    if scale:
        _centers = (nodules.nodule_center - nodules.origin) / nodules.spacing
        _rads = (nodules.nodule_size / nodules.spacing)
    return np.hstack([np.expand_dims(_rads.max(axis=1), axis=1), _centers])


@njit(cache=True)
def sphere_overlap(nodule_true, nodule_pred):
    """ Two nodules overlap volume normalized by total volume of second one.

    Args:
    - nodule_true: ndarray with information about true nodule;
    nodule_true[1:] - [z,y,x] coordinates of true nodule's center,
    nodule_true[0] - radius of true nodule;
    - nodule_pred: ndarray with information about predicted nodule;
    nodule_pred[1:] - [z,y,x] coordinates of predicted nodule's center,
    nodule_pred[0] - radius of predicted nodule;

    Returns:
    - float, normalized overlap volume.
    """
    r1, r2 = nodule_true[3], nodule_pred[3]
    pos1, pos2 = nodule_true[:3], nodule_pred[:3]

    pos1_area = 4. / 3. * np.pi * r1 ** 3
    pos2_area = 4. / 3. * np.pi * r2 ** 3

    d = np.sum((pos1 - pos2) ** 2) ** 0.5

    if d >= r1 + r2:
        return 0
    elif r1 >= d + r2:
        if r1 > 5 * r2:
            return 0
        else:
            return 1
    elif r2 >= d + r1:
        return 1

    volume = (np.pi * (r1 + r2 - d) ** 2
              * (d ** 2 + r1 * (2 * d - 3 * r1)
                 + r2 * (2 * d - 3 * r2)
                 + 6 * r1 * r2)) / (12 * d + 10e-7)
    return 2 * volume / (pos2_area + pos1_area + 10e-7)


@njit
def nodules_sets_overlap_jit(nodules_true, nodules_pred):
    """ Compute overlap matrix for two sets of nodules.

    Args:
    - nodules_true: ndarray(l, 4), contains info about centers of predicted nodules
    and theirs diameters;
    - nodules_pred: ndarray(k, 4), contains info about centers of predicted nodules
    and theirs diameters;

    Returns:
    - overlap_matrix, ndarray(l, k);
    """
    num_pred = nodules_pred.shape[0]
    num_true = nodules_true.shape[0]

    overlap_matrix = np.zeros(shape=(num_true, num_pred))
    for i in range(num_pred):
        for j in range(num_true):
            overlap_volume = sphere_overlap(nodules_true[j, :],
                                            nodules_pred[i, :])
            overlap_matrix[j, i] = overlap_volume

    return overlap_matrix
