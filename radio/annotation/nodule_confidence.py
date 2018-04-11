"""  Functions form nodules' confidences computation. """

import pandas as pd
import numpy as np


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
    cleaned = annotations.loc[:, ['coordZ', 'coordY', 'coordX', 'AccessionNumber',
                                  'DoctorID', 'NoduleID', 'DoctorConfidence']]

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
