import pandas as pd
import numpy as np

def ep(u):
    """ vectorized Epanechnikov kernel """
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1).astype(np.float)

def compute_nodule_confidence(annotations, scan_id, nodule_id=None, r=20, alpha=None,
                              weight_by_doctor=True):
    scan = annotations[annotations.AccessionNumber==scan_id]
    nodule_id = scan.NoduleID.unique() if nodule_id is None else nodule_id

    # iterize nodule_id
    nodule_id = np.array(nodule_id).reshape(-1)

    # matrix of distances
    scan_cleaned = scan.drop(labels=['annotator_1', 'annotator_2', 'annotator_3',
                             'diameter_mm', 'NoduleType', 'AccessionNumber'], axis=1)
    scan_cleaned['_key'] = 0
    pairwise = (pd
                .merge(scan_cleaned, scan_cleaned, how='inner', left_on='_key', right_on='_key', suffixes=('', '_other'))
                .drop(labels=['_key'], axis=1))
    pairwise['Distance'] = np.sqrt((pairwise.coordX - pairwise.coordX_other)**2 + (pairwise.coordY - pairwise.coordY_other)**2 +
                                   (pairwise.coordZ - pairwise.coordZ_other)**2)
    pairwise = pairwise.drop(labels=(['coord' + capital for capital in ['X', 'Y', 'Z']] +
                                     ['coord' + capital + '_other' for capital in ['X', 'Y', 'Z']]), axis=1)

    # compute kernel-weights of nodules
    nodule_ds = pairwise[[x in nodule_id for x in pairwise.NoduleID]]
    nodule_in_support = nodule_ds[nodule_ds.Distance <= r]
    nodule_in_support['weights'] = ep(nodule_in_support.Distance / r)
    nodule_in_support['weights'] *= np.maximum(nodule_in_support.DoctorID_other!=nodule_in_support.DoctorID,
                                               nodule_in_support.NoduleID_other==nodule_in_support.NoduleID)

    # compute confidences
    nodule_in_support['weighted_confs'] = (nodule_in_support['weights'] * nodule_in_support['DoctorConfidence_other'])

    # add doctor weights if needed
    if weight_by_doctor:
        nodule_in_support['weighted_confs'] *= nodule_in_support['DoctorConfidence']

    # add correction for alpha-confidences if needed
    if alpha is not None:
        # get rid of kernel weights before target-nodules and weight by alpha, 1 - alpha
        nodule_in_support.loc[nodule_in_support.NoduleID_other==nodule_in_support.NoduleID, 'weighted_confs'] *= alpha / ep(0)
        nodule_in_support.loc[nodule_in_support.NoduleID_other!=nodule_in_support.NoduleID, 'weighted_confs'] *= 1 - alpha

    confs = nodule_in_support.groupby('NoduleID').weighted_confs.sum()
    return confs
