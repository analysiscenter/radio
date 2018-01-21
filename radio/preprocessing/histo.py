""" Auxiliary functions for nodules generation """

import numpy as np


def cart_triples(*arrs):
    """ Get array of cartesian triples from three arrays.

    The order is triples is lexicographic.

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
    size : int
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


def sample_ellipsoid_region(center, axes, mult_range, size):
    """ Create a sample from `almost` uniform distribution with support given by a peel of a 3d-ellispoid.

    Parameters
    ----------
    center : tuple, list or ndarray
        center of the ellipsoid to be used for sampling.
    axes : tuple, list or ndarray
        three axes of the ellipsoid.
    mult_range : tuple, list
        range that defines the peel. E.g., mult_range = (1.0, 1.2).
        Then the peel is defined as a region, bounded from inside and outside by surfaces of two ellipsoids.
        The interior one has axes = 1.0 * axes, while the exterior one has axes = 1.2 * axes.
    size : int
        len of sample to be generated.

    Returns
    -------
    ndarray
        3d-array of shape = (size, 3) containing generated sample.
    """
    # generate uniform sample of polar and azimuthal angles
    shifted_polar = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=size)
    azimuthal = np.random.uniform(low=-np.pi, high=np.pi, size=size)

    # generate random multiplier and apply it to axes
    sample = np.asarray(axes).reshape(1, 3) * np.random.uniform(low=mult_range[0], high=mult_range[1], size=(size, 1))

    # calculate sample of points using generated sample of axes and spherical angle coords
    sample[:, 0:2] *= np.cos(shifted_polar.reshape(size, 1))
    sample[:, 0] *= np.cos(azimuthal)
    sample[:, 1] *= np.sin(azimuthal)
    sample[:, 2] *= np.sin(shifted_polar)

    # apply the shift to center
    sample += np.asarray(center)

    return sample
