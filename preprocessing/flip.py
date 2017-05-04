""" Flip CT-image """
from numba import njit


@njit(nogil=True)
def flip_patient_numba(patient, out_patient, res):
    """ Invert the order of slices """
    out_patient[:, :, :] = patient[::-1, :, :]
    return res, out_patient.shape
