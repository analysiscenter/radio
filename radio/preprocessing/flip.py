""" Flip CT-image """
from numba import njit


@njit(nogil=True)
def flip_patient_numba(patient, out_patient, res):
    """ Invert the order of slices in scan or crop.

    Parameters
    ----------
    patient : ndarray
        one item in batch (patient's scan or crop).
    out_patient : ndarray
        one item's array after inversion of z slixes.
    res : ndarray
        `skyscraper` of all  data, see _init_rebuid.

    Returns
    -------
    tuple
        (res, out_patient.shape), `skyscraper` and shape of flipped item.
    """
    out_patient[:, :, :] = patient[::-1, :, :]
    return res, out_patient.shape
