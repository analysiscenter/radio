""" Different useful functions when working with models and CTImagesMaskedBatch. """

import numpy as np
from numba import njit


def nodules_info_to_rzyx(nodules, scale=True):
    """ Transform data contained in nodules_info array to rzyx format. """
    if scale:
        _centers = (nodules.nodule_center - nodules.origin) / nodules.spacing
        _rads = (nodules.nodule_size / nodules.spacing)
    return np.hstack([np.expand_dims(_rads.max(axis=1), axis=1), _centers])


@njit(cache=True)
def sphere_overlap(nodule_true, nodule_pred):
    """ Two nodules overlap volume normalized by total volume of second one.

    Parameters
    ----------
    nodule_true : ndarray
        numpy array with information about true nodule:
        nodule_true[1:] - [z,y,x] coordinates of true nodule's center,
        nodule_true[0] - diameter of true nodule.
    nodule_pred : ndarray
        numpy array with information about predicted nodule:
        nodule_pred[1:] - [z,y,x] coordinates of predicted nodule's center,
        nodule_pred[0] - diameter of predicted nodule.

    Returns
    -------
    float
        overlap volume divided by sum of input nodules' volumes.
    """
    r1, r2 = nodule_true[0] / 2, nodule_pred[0] / 2
    pos1, pos2 = nodule_true[1:], nodule_pred[1:]

    pos1_area = 4. / 3. * np.pi * r1 ** 3
    pos2_area = 4. / 3. * np.pi * r2 ** 3

    d = np.sum((pos1 - pos2) ** 2) ** 0.5

    if d >= r1 + r2:
        return 0
    elif r1 >= d + r2:
        if r1 > 5 * r2:
            return 0
        else:
            return 1
    elif r2 >= d + r1:
        return 1

    volume = (np.pi * (r1 + r2 - d) ** 2
              * (d ** 2 + r1 * (2 * d - 3 * r1)
                 + r2 * (2 * d - 3 * r2)
                 + 6 * r1 * r2)) / (12 * d + 10e-7)
    return 2 * volume / (pos2_area + pos1_area + 10e-7)


@njit
def nodules_sets_overlap_jit(nodules_true, nodules_pred):
    """ Compute overlap matrix for two sets of nodules.

    Parameters
    ----------
    nodules_true : ndarray(l, 4)
        numpy array containing info about centers of target nodules and theirs diameters.
    nodules_pred : ndarray(k, 4)
        numpy array containing info about centers of predicted nodules and theirs diameters.

    Returns
    -------
    ndarray(l, k)
        overlap matrix for two sets of nodules.
    """
    num_pred = nodules_pred.shape[0]
    num_true = nodules_true.shape[0]

    overlap_matrix = np.zeros(shape=(num_true, num_pred))
    for i in range(num_pred):
        for j in range(num_true):
            overlap_volume = sphere_overlap(nodules_true[j, :],
                                            nodules_pred[i, :])
            overlap_matrix[j, i] = overlap_volume

    return overlap_matrix


def _create_overlap_index(overlap_matrix):
    """ Get indices of nodules that overlaps using overlap_matrix. """
    argmax_ov = overlap_matrix.argmax(axis=1)
    max_ov = overlap_matrix.max(axis=1).astype(np.bool)
    return max_ov, argmax_ov


def overlap_nodules(batch, nodules_true, nodules_pred):
    """ Get info about overlap between true and predicted nodules.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        input batch
    nodules_true : numpy record array
        numpy record array of type CTImagesMaskedBatch.nodules_dtype
        with true nodules.
    nodules_pred : numpy record array
        numpy record array of type CTImagesMaskedBatch.nodules_dtype
        with predicted nodules.

    Returns
    -------
    dict
        {'true_stats': pd.DataFrame, 'pred_stats': pd.DataFrame}
    """
    true_df = (
        batch
        .nodules_to_df(nodules_true)
        .assign(diam=lambda df: np.max(df[['diamZ', 'diamY', 'diamX']], axis=1))
    )

    pred_df = (
        batch
        .nodules_to_df(nodules_pred)
        .assign(diam=lambda df: np.max(df[['diamZ', 'diamY', 'diamX']], axis=1))
    )

    true_out, pred_out = [], []
    true_gr, pred_gr = true_df.groupby('source_id'), pred_df.groupby('source_id')
    for group_name in {**true_gr.groups, **pred_gr.groups}:
        try:
            nods_true = true_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX',
                                                              'nodule_id', 'confidence']]
        except KeyError:
            nods_pred = pred_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX',
                                                              'nodule_id', 'confidence']]
            nods_pred.loc[:, 'overlap_index'] = np.nan
            nods_pred.loc[:, 'source_id'] = group_name
            nods_pred = nods_pred.set_index('nodule_id')
            pred_out.append(nods_pred)
            continue
        try:
            nods_pred = pred_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX',
                                                              'nodule_id', 'confidence']]
        except KeyError:
            nods_true = true_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX',
                                                              'nodule_id', 'confidence']]
            nods_true.loc[:, 'overlap_index'] = np.nan
            nods_true.loc[:, 'source_id'] = group_name
            nods_true = nods_true.set_index('nodule_id')
            true_out.append(nods_true)
            continue

        nods_true = nods_true.set_index('nodule_id').loc[:, ['diam', 'locZ', 'locY',
                                                             'locX', 'confidence']]
        nods_pred = nods_pred.set_index('nodule_id').loc[:, ['diam', 'locZ', 'locY',
                                                             'locX', 'confidence']]

        overlap_matrix = nodules_sets_overlap_jit(nods_true.values[:,:-1], nods_pred.values[:,:-1])

        ov_mask_true, ov_ind_true = _create_overlap_index(overlap_matrix)
        ov_mask_pred, ov_ind_pred = _create_overlap_index(overlap_matrix.T)

        nods_true = nods_true.assign(overlap_index=lambda df: df.index)
        nods_true.loc[ov_mask_true, 'overlap_index'] = nods_pred.index[ov_ind_true[ov_mask_true]]
        nods_true.loc[np.logical_not(ov_mask_true), 'overlap_index'] = np.nan
        nods_true.loc[:, 'source_id'] = group_name

        nods_pred = nods_pred.assign(overlap_index=lambda df: df.index)
        nods_pred.loc[ov_mask_pred, 'overlap_index'] = nods_true.index[ov_ind_pred[ov_mask_pred]]
        nods_pred.loc[np.logical_not(ov_mask_pred), 'overlap_index'] = np.nan
        nods_pred.loc[:, 'source_id'] = group_name

        true_out.append(nods_true)
        pred_out.append(nods_pred)

    return {'true_stats': true_out, 'pred_stats': pred_out}
