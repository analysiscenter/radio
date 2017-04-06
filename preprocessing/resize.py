"""
Module with auxillary
    jit-compiled functions
    for resize of
    DICOM-scans
"""

from numba import jit
import scipy.ndimage


@jit('void(double[:,:,:], int64, int64, int64, int64, int64, double[:,:,:], int64)',
     nogil=True)
def resize_patient_numba(chunk, start_from, end_from, num_x_new,          # pylint: disable=too-many-arguments
                         num_y_new, num_slices_new, order, res, start_to):
    """
    resizes 3d-scan for one patient
        args
        -chunk: skyscraper from which the patient data is taken
        - start_from: first floor for patient from chunk
        - end-from: last floor for patient from chunk
        - num_x_new: needed x-dimension
        - num_y_new: needed y-dimension
        - num_slices_new: needed number of slices
        - order: order of interpolation
        - res: skyscraper where to put the resized patient
        - start_to: first floor for resized patient in
    """
    # define resize factor
    res_factor = [num_slices_new / float((end_from - start_from)), num_y_new / float(chunk.shape[1]),
                  num_x_new / float(chunk.shape[2])]

    # perform resizing anf put the result into res[satrt_to:]
    res[start_to:start_to + num_slices_new] = scipy.ndimage.interpolation.zoom(
        chunk[start_from:end_from, :, :], res_factor, order=order)
