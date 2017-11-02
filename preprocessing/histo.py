""" Auxiliary functions for nodules generation """

import numpy as np


def cart_triples(*arrs):
    """ Get array of element-wise triples from sequence of 3 arrays

    Match elements of arrays at each position (first-with-first) into triples.

    Parameters
    ----------
    arrs : tuple, list or ndarray
           Any sequence of 3d ndarrays.

    Returns
    -------
    ndarray
            2d-array of triples (array1_item_n,array2_item_n,array3_item_n)
    """
    res = np.transpose(np.stack(np.meshgrid(*arrs)), axes=(2, 1, 3, 0))
    return res.reshape(-1, 3)


def sample_histo3d(histo, size):
    """ Create a sample of size=size from distribution represented by 3d-histogram

    Parameters
    ----------
    histo : tuple
            (bins, edges) of np.histogram(). `bins` is a 3d-array, number of points in a specific cube.
            `edges` is a list of 3 arrays of len = (nbins_in_dimension + 1),
            represents bounds of bins' boxes.
    size :  int
            length of sample to be generated

    Returns
    -------
    ndarray
            3d-array of shape = (size, 3), contains samples from histo-distribution
    """
    # infer probabilities of bins, sample number of bins according to these probs
    probs = (histo[0] / np.sum(histo[0])).reshape(-1)
    bin_nums = np.random.choice(np.arange(histo[0].size), p=probs, size=size)

    # lower and upper bounds of boxes
    l_all = cart_triples(histo[1][0][:-1], histo[1][1][:-1], histo[1][2][:-1])
    h_all = cart_triples(histo[1][0][1:], histo[1][1][1:], histo[1][2][1:])

    # uniformly generate samples from selected boxes
    l, h = l_all[bin_nums], h_all[bin_nums]
    return np.random.uniform(low=l, high=h)
