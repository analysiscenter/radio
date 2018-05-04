""" Contains common functions """
import numpy as np
from scipy.ndimage import measurements


def binarize(mask, threshold=.5):
    """ Create a binary mask from probabilities with a given threshold.

    Parameters
    ----------
    mask : np.array
        input mask with probabilities
    threshold : float
        where probability is above the threshold, the output mask will have 1, otherwise 0.

    Returns
    -------
    np.array
        binary mask of the same shape as the input mask
    """
    _mask = np.asarray(mask)
    if _mask.dtype == np.bool:
        return mask
    return np.where(_mask >= threshold, 1, 0).astype(np.bool)


def get_nodules(mask):
    """ Find nodules as connected components in the input mask.

    Parameters
    ----------
    mask : np.array
        Binary mask

    Returns
    -------
    np.array or None
        An array of coords of the nodules found
    """
    connected_array, num_components = measurements.label(mask, output=None)
    if num_components == 0:
        return None
    nodules = []
    for i in range(1, num_components + 1):
        coords = [slice(np.min(c), np.max(c) + 1) for c in np.where(connected_array == i)]
        nodules.append(coords)
    return nodules



