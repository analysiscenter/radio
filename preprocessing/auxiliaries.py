"""
Module with auxillary
    jit-compiled functions
    for DICOM-scans
    preprocessing
"""

import numpy as np
from numba import jit
from skimage import measure, morphology
import scipy.ndimage


@jit(nogil=True)
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return -999999


@jit('void(double[:,:,:], double[:], double[:], int64, int64, int64, double[:,:,:], int64)',
     nogil=True)
def resize_chunk_numba(concatted_chunk, l_bds, u_bds,
                       num_slices_new, num_x_new, num_y_new, res, st_ind):
    for i in np.arange(len(l_bds)):
        # define resize factor
        res_factor = [num_slices_new / float((u_bds[i] - l_bds[i])), num_x_new / float(concatted_chunk.shape[1]),
                      num_y_new / float(concatted_chunk.shape[2])]

        # perform resizeing
        res[st_ind + i * num_slices_new: st_ind + (i + 1) * num_slices_new, :, :] = scipy.ndimage.interpolation.zoom(
            concatted_chunk[l_bds[i]:u_bds[i], :, :], res_factor)

        """
        remember, result is put into res-argument
            starting from st_ind position
        so, no return 
        """


@jit('void(double[:,:,:], int64, int64, int64, int64, int64, double[:,:,:], int64)',
     nogil=True)
def resize_patient_numba(chunk, start_from, end_from, num_slices_new,
                         num_x_new, num_y_new, res, start_to):
    # define resize factor
    res_factor = [num_slices_new / float((end_from - start_from)), num_x_new / float(chunk.shape[1]),
                  num_y_new / float(chunk.shape[2])]

    # perform resizing anf put the result into res[satrt_to:]
    res[start_to:start_to + num_slices_new] = scipy.ndimage.interpolation.zoom(
        chunk[start_from:end_from, :, :], res_factor)


# segmentation of a patient sliced from skyscraper
@jit('void(double[:,:,:], int64, int64, double[:,:,:], int64, int64)',
     nogil=True)
def get_filter_patient(chunk, start_from, end_from, res, start_to, erosion_radius):

    # slice the patient out from the skyscraper
    # we use view for simplification
    data = chunk[start_from:end_from, :, :]

    binary_image = np.array(data > -320, dtype=np.int8) + 1

    # 3d connected components
    labels = measure.label(binary_image)

    bin_shape = binary_image.shape

    # create np.array of unique labels of corner pixels
    corner_labs = np.zeros(0)

    for zi in range(bin_shape[0]):
        corner_labs = np.append(corner_labs, labels[zi, 0, 0])
        corner_labs = np.append(corner_labs, labels[zi, bin_shape[1] - 1, 0])
        corner_labs = np.append(
            corner_labs, labels[zi, bin_shape[1] - 1, bin_shape[2] - 1])
        corner_labs = np.append(corner_labs, labels[zi, 0, bin_shape[2] - 1])

    bk_labs = np.unique(corner_labs)

    # Fill the air around the person
    for background_label in bk_labs:
        binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    # For every slice we determine the largest solid structure

    # seems like to this point binary image is what it should be
    # return binary_image

    for i, axial_slice in enumerate(binary_image):

        axial_slice = axial_slice - 1

        '''
        in each axial slice lungs and air are labelled as 0 
        everything else has label = 1

        look for and enumerate connected components in axial_slice
        '''
        # 2d connected components
        labeling = measure.label(axial_slice)

        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None:  # This slice contains some lung
            binary_image[i][labeling != l_max] = 1

    # return binary_image
    # it's all fine here

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body

    # stil fine
    # return binary_image

    # again, 3d connected components
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    # slightly erode the filter
    # to get rid of lungs' boundaries

    # return binary_image

    selem = morphology.disk(erosion_radius)

    # put the filter into the result
    for i in range(end_from - start_from):
        res[start_to + i, :,
            :] = morphology.binary_erosion(binary_image[i, :, :], selem)
