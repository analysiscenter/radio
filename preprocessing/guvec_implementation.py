"""
guvectorized
    segmentation and sizes-unification

not working properly!
"""

import numpy as np
from numba import guvectorize, int64, float64
from skimage import measure, morphology
import scipy.ndimage


"""
OK, let's try parallelize everything
    through guvectorize
We test the approach on resize
If the whole thing works
    we will use the approach hereafter

args:
    bounds = [lower_bound_from, upper_bound_from,
            lower_bound_to, upper_bound_to]: indices where the patient 
            can be found and where he should be put

    num_slices_new, num_x_new, num_y_new - new shapes of the 3d-scan

    batch_from: from where the patient is sliced
    batch_to: to where the resized patient is put
"""


@guvectorize([int64[:], int64[:], float64[:, :, :], float64[:, :, :]],
             '(l),(u),(m,n,p)->(r,q,s)')
def resize_patient(bounds, new_shapes, batch_from, batch_to):
    num_slices_new, num_x_new, num_y_new = new_shapes[
        0], new_shapes[1], new_shapes[2]
    res_factor = [num_slices_new / float(bounds[1] - bounds[0]), num_x_new / float(batch_from.shape[1]),
                  num_y_new / float(batch_from.shape[2])]
    batch_to[bounds[2]: bounds[3], :, :] = scipy.ndimage.interpolation.zoom(
        batch_from[bounds[0]:bounds[1], :, :], res_factor)

    # again, no result
    # this is the only way guvectorize works


"""
#SO! no good
    numba's layouts do not currently support uknown output-dimensions

So, with this particular example of resize 
    we'll have to stick with multithreaded version   




"""


# try guvectorize segmentation!
@guvectorize(['void(int64[:], int64[:], float64[:,:,:], float64[:,:,:], float64[:])'], '(l),(r),(m,n,p),(m,n,p)->()', nopython=False,
             target='parallel', forceobj=True)
def get_patient_segment(bounds, params, batch_from, batch_to, fake):
    #print('HELLO AGAIN')
    pat_lb_from = bounds[0]
    pat_ub_from = bounds[1]
    pat_lb_to = bounds[2]
    pat_ub_to = bounds[3]
    erosion_radius = params[0]

    # we use view for simplification
    data = batch_from[pat_lb_from:pat_ub_from, :, :]

    binary_image = np.array(data > -320, dtype=np.int8) + 1
    #binary_image = np.zeros_like(data)
    labels = measure.label(binary_image)

    bin_shape = binary_image.shape
    # create list of unique labels of corner pixels
    corner_labs = []

    for zi in range(bin_shape[0]):
        corner_labs.append(labels[zi, 0, 0])
        corner_labs.append(labels[zi, bin_shape[1] - 1, 0])
        corner_labs.append(labels[zi, bin_shape[1] - 1, bin_shape[2] - 1])
        corner_labs.append(labels[zi, 0, bin_shape[2] - 1])

    """
    corner_labs += [labels[zi, 0, 0] for zi in range(0, bin_shape[0], 1)]
    corner_labs += [labels[zi, bin_shape[1]-1, 0] for zi in range(0, bin_shape[0], 1)]
    corner_labs += [labels[zi, bin_shape[1]-1, bin_shape[2]-1] for zi in range(0, bin_shape[0], 1)]
    corner_labs += [labels[zi, 0, bin_shape[2]-1] for zi in range(0, bin_shape[0], 1)]
    """

    bk_labs = list(set(corner_labs))

    # Fill the air around the person
    for background_label in bk_labs:
        binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    # For every slice we determine the largest solid structure
    # print('here')

    for i, axial_slice in enumerate(binary_image):

        axial_slice = axial_slice - 1

        '''
        in each axial slice lungs and air are labelled as 0 
        everything else has label = 1

        look for and enumerate connected components in axial_slice
        '''
        labeling = measure.label(axial_slice)

        vals, counts = np.unique(labeling, return_counts=True)
        bg = 0

        counts = counts[vals != bg]
        vals = vals[vals != bg]
        ##########

        l_max = vals[np.argmax(counts)]
        #l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None:  # This slice contains some lung
            binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    # slightly erode the filter
    # to get rid of lungs' boundaries

    selem = morphology.disk(erosion_radius)

    for i in range(pat_lb_to, pat_lb_to + binary_image.shape[0]):
        batch_to[i, :, :] = morphology.binary_erosion(
            binary_image[i, :, :], selem)


"""
currently forceobj = True and 
    target = 'parallel'
DO NOT coexist 
    that is, the execution of the function goes only
    forever

wait till it's fixed, after that one will
    be able to use this for instant multithreading
    in Batch class
"""
