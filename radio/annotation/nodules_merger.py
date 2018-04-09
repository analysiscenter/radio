""" Functions for mapping sets of overlapping nodules into one nodule. """

import numpy as np
import pandas as pd
from scipy import stats
from .parser import generate_index
from ..models.utils import sphere_overlap


def assign_nodules_group_index(nodules):
    """ Add column with name 'GroupNoduleID' containing index of group of overlapping nodules.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules locations and centers.

    Returns
    -------
    pandas DataFrame
    """
    overlap_groups = {}
    for _, row_l in nodules.iterrows():
        overlap_indices = []
        for nodule_r, row_r in nodules.iterrows():
            al = row_l.loc[['diameter_mm', 'coordZ', 'coordY', 'coordX']].values.astype(np.float)
            ar = row_r.loc[['diameter_mm', 'coordZ', 'coordY', 'coordX']].values.astype(np.float)

            if sphere_overlap(al, ar) > 0:
                overlap_indices.append(nodule_r)

        if not any(nodule_id in overlap_groups for nodule_id in overlap_indices):
            index = generate_index()
        else:
            nodules_list = [nodule_id for nodule_id in overlap_indices if nodule_id in overlap_groups]
            index = overlap_groups[nodules_list[0]]

        for nodule_id in overlap_indices:
            overlap_groups[nodule_id] = index

    return nodules.assign(GroupNoduleID=pd.Series(overlap_groups))


def get_diameter_by_sigma(sigma, proba):
    """ Get diameter of nodule given sigma of normal distribution and probability of diameter coverage area.

    Transforms sigma parameter of normal distribution corresponding to cancerous nodule
    to its diameter using probability of diameter coverage area.

    Parameters
    ----------
    sigma : float
        square root of normal distribution variance.
    proba : float
        probability of diameter coverage area.

    Returns
    -------
    float
        equivalent diameter.
    """
    return 2 * sigma * stats.norm.ppf((1 + proba) / 2)  # pylint: disable=no-member


def get_sigma_by_diameter(diameter, proba):
    """ Get sigma of normal distribtuion by diameter of nodule and probability of diameter coverage area.

    Parameters
    ----------
    diameter : float
        diameter of nodule.
    proba : float
        probability of diameter coverage area.

    Returns
    -------
    float
        equivalent normal distribution's sigma parameter.
    """
    return diameter / (2 * stats.norm.ppf((1 + proba) / 2))  # pylint: disable=no-member


def approximate_gaussians(confidence_array, mean_array, variance_array):
    """ Approximate gaussians with given parameters with one gaussian.

    Approximation is performed via minimization of Kullback-Leibler
    divergence KL(sum_{j} w_j N_{mu_j, sigma_j} || N_{mu, sigma}).

    Parameters
    ----------
    confidence_array : ndarray(num_gaussians)
        confidence values for gaussians.
    mean_array : ndarray(num_gaussians, 3)
        (z,y,x) mean values for input gaussians.
    variance_array : ndarray(num_gaussians)
        (z,y,x) variances for input gaussians.

    Returns
    -------
    tuple(ndarray(3), ndarray(3))
        mean and sigma for covering gaussian.
    """
    delimiter = np.sum(confidence_array)
    mu = np.sum(mean_array.T * confidence_array, axis=1) / delimiter
    sigma = np.sqrt(np.sum((variance_array + (mean_array - mu) ** 2).T
                           * confidence_array, axis=1) / delimiter)
    return mu, sigma


def compute_group_coords_and_diameter(nodules, proba=0.8):
    """ Get coordinates of center and diameter of nodules united in group.

    For each group of overlapping nodules computes equivalent diameter and
    coordinates of center. Preserves 'confidence' and 'AccessionNumber'
    columns from source nodules dataframe. Note, that this columns
    are considered to contain same values within group.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules location and sizes.
    proba : float
        float value from [0, 1] interval. Probability of diameter coverage area
        for equivalent normal distribution.

    Returns
    -------
    pandas DataFrame
        dataframe with information about equivalent locations and diameters of
        groups of overlapping nodules.
    """
    num_nodules = nodules.shape[0]
    confidence_array = np.zeros(num_nodules, dtype=np.float64)
    mean_array = np.zeros((num_nodules, 3), dtype=np.float64)
    variance_array = np.zeros(num_nodules, dtype=np.float64)
    for i, (_, row) in enumerate(nodules.iterrows()):
        mean_array[i, :] = np.array((row['coordZ'], row['coordY'], row['coordX']))
        variance_array[i] = get_sigma_by_diameter(row['diameter_mm'], proba=proba) ** 2
        confidence_array[i] = row['confidence']

    variance_array = np.tile(variance_array[:, np.newaxis], (1, 3))
    approx_mean, approx_sigma = approximate_gaussians(confidence_array, mean_array, variance_array)
    return  pd.Series({'coordZ': approx_mean[0], 'coordY': approx_mean[1],
                       'coordX': approx_mean[2], 'confidence': confidence_array.max(),
                       'AccessionNumber': nodules.AccessionNumber.iloc[0],
                       'diameter_mm': get_diameter_by_sigma(approx_sigma, proba=proba)[0]})


def get_nodules_groups(nodules, proba=0.8):
    """ Unite overlapping nodules in groups and compute equivalent diameter and locations of groups.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules.
    proba : float
        float from [0, 1] interval. Probability of diameter coverage area of
        equivalent normal distribution.

    Returns
    -------
    pandas DataFrame
        dataframe with information about overlapping nodules groups centers
        locations and diameters.
    """
    new_nodules = (
        nodules
        .set_index(['AccessionNumber', 'NoduleID'])
        .groupby(level=0)
        .apply(assign_nodules_group_index)
        .reset_index()
        .groupby('GroupNoduleID')
        .apply(compute_group_coords_and_diameter, proba=proba)
        .reset_index()
    )
    return new_nodules
