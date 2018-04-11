"""  Functions form nodules' confidences computation. """

import pandas as pd
import numpy as np
from numba import njit



def ep(u):
    """ Vectorized Epanechnikov kernel.

    Parameters
    ----------
    u : ndarray
        input array of distances.

    Return
    ------
    ndarray
        array of ep(input array-items).
    """
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1).astype(np.float)


@njit
def epanechnikov_kernel_numba(x):
    return 0.75 * (1 - u ** 2) * (x <= 1).astype(np.float)


@njit
def get_distance_matrix_numba(coords):
    num_items = coords.shape[0]
    distance_matrix = np.zeros(shape=(num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            distance_matrix[i, j] = np.sqrt(np.sum((coords[i, :] - coords[j, :]) ** 2))
    return distance_matrix

@njit
def get_nodules_doctors_mask_numba(nodules_doctors):
    num_items = nodules_doctors.shape[0]
    confidence_matrix = np.ones(shape=(num_items, num_itemss), dtype=int)
    for i in range(num_items):
        for j in range(num_items):
            if i != j:
                confidence_matrix[i, j] = int(not nodules_doctors[i] == nodules_doctors[j])
    return confidence_matrix


def compute_nodule_confidence(nodules, r=20, alpha=None, weight_by_doctor=True):
    coords = nodules.loc[:, ['coordZ', 'coordY', 'coordX']].values
    doctors_confidences = nodules.loc[:, 'DoctorConfidence'].values
    nodules_doctors = nodules.loc[:, 'DoctorID'].astype(int).values

    distance_matrix = epanechnikov_kernel_numba(get_distance_matrix_numba(coords))
    confidences = distance_matrix * get_nodules_doctors_mask_numba(nodules_doctors)



def compute_nodule_confidence(annotations, r=20, alpha=None, weight_by_doctor=True):
    """ Compute nodule confidence using annotations-df; put confidences into a new column
    'NoduleConfidence'.

    Parameters
    ----------
    annotations : pd.DataFrame
        input df with annotations.
    r : float
        radius of kernel-support.
    alpha : float or None
        weight of target-nodule's doctor in weighted sum.
    weight_by_doctor : bool
        whether the weighted sum should be weighted by target-nodule's doctor.

    Return
    ------
    pd.DataFrame
        annotations-dataframe with added 'NoduleConfidence'-column.
    """
    # matrix of distances between nodules from the same scan
    cleaned = annotations.loc[:, ['coordZ', 'coordY', 'coordX', 'AccessionNumber', 'NoduleID']]

    pairwise = pd.merge(cleaned, cleaned, how='inner', left_on='AccessionNumber',
                        right_on='AccessionNumber', suffixes=('', '_other'))

    pairwise['Distance'] = np.sqrt((pairwise.coordX - pairwise.coordX_other) ** 2
                                   + (pairwise.coordY - pairwise.coordY_other)** 2
                                   + (pairwise.coordZ - pairwise.coordZ_other) ** 2)

    pairwise = pairwise.drop(labels=['coordZ', 'coordY', 'coordX', 'coordZ_other',
                                     'coordY_other', 'coordX_other'], axis=1)

    # compute kernel-weights of nodules
    pairwise = pairwise[pairwise.Distance <= r]
    pairwise['weights'] = ep(pairwise.Distance / r)
    pairwise['weights'] *= np.maximum(pairwise.DoctorID_other != pairwise.DoctorID,
                                      pairwise.NoduleID_other == pairwise.NoduleID)

    # compute confidences
    pairwise['weighted_confs'] = (pairwise['weights'] * pairwise['DoctorConfidence_other'])

    # add doctor weights if needed
    if weight_by_doctor:
        pairwise['weighted_confs'] *= pairwise['DoctorConfidence']

    # add correction for alpha-confidences if needed
    if alpha is not None:
        # get rid of kernel weights before target-nodules and weight by alpha, 1 - alpha
        pairwise.loc[pairwise.NoduleID_other == pairwise.NoduleID, 'weighted_confs'] *= alpha / ep(0)
        pairwise.loc[pairwise.NoduleID_other != pairwise.NoduleID, 'weighted_confs'] *= 1 - alpha

    confs = pairwise.groupby('NoduleID').weighted_confs.sum()
    confs = pd.DataFrame(confs)
    confs.rename({'weighted_confs': 'NoduleConfidence'}, axis=1, inplace=True)
    return pd.merge(annotations, confs, left_on='NoduleID', right_index=True)
