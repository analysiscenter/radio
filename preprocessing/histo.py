""" Auxiliary functions for nodules generation """

import numpy as np


def cart_triples(*arrs):
    """ Get array of cartesian triples in lexicographical order given sequence of 3 arrays

    Args:
        arrs: sequence of 3 arrays
    Return:
        2darray of triples of shape = (len(arr[0]) * len(arr[1]) * len(arr[2]), 3)
    """
    res = np.transpose(np.stack(np.meshgrid(*arrs)), axes=(2, 1, 3, 0))
    return res.reshape(-1, 3)


def sample_histo3d(histo, size):
    """ Create a sample of size=size from distribution represented by 3d-histogram

    Args:
        histo: 3d-histogram given by tuple (bins, edges) (np.histogramdd format). Item bins
            is a 3d-array and stands for number of points in a specific cube. Edges is a list of
            3 arrays of len = (nbins_in_dimension + 1), represents bounds of bins' boxes.
        size: len of sample to be generated

    Return:
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
