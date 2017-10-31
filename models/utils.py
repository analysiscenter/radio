import numpy as np
import pandas as pd
from numba import njit


@njit(nogil=True)
def create_mask_jit(masks, start, end):
    """ Jit-decorated function for fast computation of masks by regression data.

    This function is usually called inside create_mask_reg function.
    """
    num_items = start.shape[0]
    for i in range(num_items):
        masks[i,
              start[i, 0]: end[i, 0],
              start[i, 1]: end[i, 1],
              start[i, 2]: end[i, 2]] = 1.
    return masks


def create_mask_reg(centers, sizes, probs, crop_shape, threshold):
    """ Create mask by data contained in predictions of regression model. """
    n_items = centers.shape[0]
    masks_array = np.zeros(shape=(n_items, *crop_shape), dtype=np.float)
    _crop_shape = np.asarray(crop_shape)

    start_pixels = np.rint(np.clip(centers - sizes / 2, 0, 1) * _crop_shape).astype(np.int)
    end_pixels = np.rint(np.clip(centers + sizes / 2, 0, 1) * _crop_shape).astype(np.int)
    positions = np.array([p > threshold for p in probs])

    masks_array[positions, ...] = create_mask_jit(masks_array[positions, ...],
                                                  start_pixels[positions, ...],
                                                  end_pixels[positions, ...])
    return masks_array


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
        nodule_true[0] - radius of true nodule.
    nodule_pred : ndarray
        numpy array with information about predicted nodule:
        nodule_pred[1:] - [z,y,x] coordinates of predicted nodule's center,
        nodule_pred[0] - radius of predicted nodule;

    Returns
    -------
    float
        overlap volume divided by sum of input nodules' volumes.
    """
    r1, r2 = nodule_true[3], nodule_pred[3]
    pos1, pos2 = nodule_true[:3], nodule_pred[:3]

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
        numpy array containing info about centers of predicted nodules and theirs diameters;

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


def unpack_component(batch, model=None, component='images', dim_ordering='channels_last'):
    """ Basic way for unpacking 'images' or 'masks' from batch.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        batch to unpack.
    model : dataset.models.BaseModel
        model where the data from batch will be fed. Is not used here.
        Required for compatibility with dataset.Pipeline.train_model
        and dataset.Pipeline.predict_model interfaces.
    component : str
        component to unpack, can be 'images' or 'masks'.
    dim_ordering : str
        can be 'channels_last' or 'channels_first'. Reflects where to put
        channels dimension: right after batch dimension or after all spatial axes.

    Returns
    -------
    ndarray(batch_size, zdim, ydim, xdim, 1)
        unpacked 'images' or 'masks' component of batch as numpy array.

    Raises
    ------
    AttributeError
        if argument component is not 'images' or 'masks'.
    """
    if component not in ('masks', 'images'):
        raise AttributeError("Component must be 'images' or 'masks'")
    x = np.stack([batch.get(i, component) for i in range(len(batch))])
    if dim_ordering == 'channels_last':
        x = x[..., np.newaxis]
    elif dim_ordering == 'channels_first':
        x = x[:, np.newaxis, ...]
    return x


def unpack_seg(batch, model=None, dim_ordering='channels_last'):
    """ Unpack data from batch in format suitable for segmentation task.

    Parameters
    ----------
    self : CTImagesMaskedBatch
        batch to unpack.
    model : dataset.models.BaseModel
        model where the data from batch will be fed. Is not used here.
        Required for compatibility with dataset.Pipeline.train_model
        and dataset.Pipeline.predict_model interfaces.
    dim_ordering : str
        can be 'channels_last' or 'channels_first'. Reflects where to put
        channels dimension: right after batch dimension or after all spatial axes.

    Returns:
    dict
        {'x': images_array, 'y': labels_array}

    NOTE
    ----
    'dim_ordering' argument reflects where to put '1'
    for channels dimension both for images and masks.
    """
    return {'x': unpack_component(batch, model, 'images', dim_ordering),
            'y': unpack_component(batch, model, 'masks', dim_ordering)}


def unpack_clf(batch, model, threshold=10, dim_ordering='channels_last'):
    """ Unpack data from batch in format suitable for classification task.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        batch to unpack.
    model : dataset.models.BaseModel
        model where the data from batch will be fed. Is not used here.
        Required for compatibility with dataset.Pipeline.train_model
        and dataset.Pipeline.predict_model interface.
    threshold : int
        minimum number of '1' pixels in mask to consider it cancerous.
    dim_ordering : str
        can be 'channels_last' or 'channels_first'. Reflects where to put
        channels dimension: right after batch dimension or after all spatial axes.

    Returns:
    - dict
        {'x': images_array, 'y': labels_array}

    NOTE
    ----
    'dim_ordering' argument reflects where to put '1' for channels dimension.
    """
    masks_labels = np.asarray([batch.get(i, 'masks').sum() > threshold
                               for i in range(len(batch))], dtype=np.int)

    return {'x': unpack_component(batch, model, 'images', dim_ordering),
            'y': masks_labels[:, np.newaxis]}


def unpack_reg(batch, model, threshold=10, dim_ordering='channels_last'):
    """ Unpack data from batch in format suitable for regression task.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        batch to unpack
    model : dataset.models.BaseModel
        model where the data from batch will be fed. Is not used here.
        Required for compatibility with dataset.Pipeline.train_model
        and dataset.Pipeline.predict_model interface.
    threshold : int
        minimum number of '1' pixels in mask to consider it cancerous.

    Returns:
    dict
        {'x': images_array, 'y': y_regression_array}

    NOTE
    ----
    'dim_ordering' argument reflects where to put '1' for channels dimension.

    TODO Need more testing.

    """

    nods = batch.nodules

    sizes = np.zeros(shape=(len(batch), 3), dtype=np.float)
    centers = np.zeros(shape=(len(batch), 3), dtype=np.float)

    for patient_pos, _ in enumerate(batch.indices):
        pat_nodules = nods[nods.patient_pos == patient_pos]

        if len(pat_nodules) == 0:
            continue

        mask_nod_indices = pat_nodules.nodule_size.max(axis=1).argmax()

        nodule_sizes = (pat_nodules.nodule_size / batch.spacing[patient_pos, :]
                        / batch.images_shape[patient_pos, :])

        nodule_centers = (pat_nodules.nodule_center / batch.spacing[patient_pos, :]
                          / batch.images_shape[patient_pos, :])

        sizes[patient_pos, :] = nodule_sizes[mask_nod_indices, :]
        centers[patient_pos, :] = nodule_centers[mask_nod_indices, :]

    clf_dict = unpack_clf(batch, model, threshold, dim_ordering)
    x, labels = clf_dict['x'], clf_dict['y']
    y_regression_array = np.concatenate([centers, sizes, labels], axis=1)

    return {'x': x, 'y': y_regression_array}


def test_on_dataset(batch, model_name, unpacker, batch_size, period, **kwargs):
    """ Compute metrics of model on dataset.

    Parameters
    ----------
    model_name : str
        name of model.
    unpacker : callable
        function that will be used  for unpacking data from test_pipeline batches.
    batch_size : int
        size of batch when making predictions.
    period : int
        frequency of test_on_dataset runs.
    kwargs : dict
        these parameters will be ignored.

    Returns
    -------
    CTImagesMaskedBatch
        unchanged input batch.
    """
    if batch.pipeline is None:
        return batch

    if batch.pipeline.config is not None:
        train_iter = batch.pipeline.get_variable('iter')

        metrics = batch.pipeline.config.get('metrics', ())
        test_pipeline = batch.pipeline.config.get('test_pipeline', None)
        test_pipeline.reset_iter()

        test_metrics = batch.pipeline.get_variable('test_metrics')

    if len(metrics) and (train_iter % period == 0):
        _model = batch.get_model_by_name(model_name)
        ds_metrics_list = []
        for batch in test_pipeline.gen_batch(batch_size):
            feed_dict = unpacker(batch, _model, **kwargs)

            y_pred = _model.predict(x=feed_dict.get('x', None))
            y_true = feed_dict.get('y', None)

            extend_data = {m.__name__: m(y_true, y_pred) for m in metrics}

            ds_metrics_list.append(extend_data)

        ds_metrics = pd.DataFrame(ds_metrics_list).mean()
        batch.pipeline.update_variable('test_metrics', value=ds_metrics.to_dict(), mode='a')
    else:
        if len(test_metrics) == 0:
            value = {m.__name__: np.nan for m in metrics}
        else:
            value = test_metrics[-2]
        batch.pipeline.update_variable('test_metrics', value=value, mode='a')
    batch.pipeline.set_variable('iter', train_iter + 1)
    return None


def _create_overlap_index(overlap_matrix):
    """ Get indices of nodules that overlaps using overlap_matrix. """
    argmax_ov = overlap_matrix.argmax(axis=1)
    max_ov = overlap_matrix.max(axis=1).astype(np.bool)
    return max_ov, argmax_ov


def overlap_true_pred_nodules(batch):
    """ Accumulate info about overlap between true and predicted nodules in pipeline vars. """
    ppl_nodules_true = batch.pipeline.get_variable('nodules_true', init=list)
    ppl_nodules_pred = batch.pipeline.get_variable('nodules_pred', init=list)

    batch_nodules_true = batch.nodules
    batch.fetch_nodules_from_mask()
    batch_nodules_pred = batch.nodules

    true_df = batch.nodules_to_df(batch_nodules_true).set_index('nodule_id')
    true_df = true_df.assign(diam=lambda df: np.max(df.iloc[:, [4, 5, 6]], axis=1))

    pred_df = batch.nodules_to_df(batch_nodules_pred).set_index('nodule_id')
    pred_df = pred_df.assign(diam=lambda df: np.max(df.iloc[:, [4, 5, 6]], axis=1))

    true_out, pred_out = [], []
    true_gr, pred_gr = true_df.groupby('source_id'), pred_df.groupby('source_id')
    for group_name in {**true_gr.groups, **pred_gr.groups}:
        try:
            nods_true = true_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX']]
        except KeyError:
            nods_pred = pred_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX']]
            pred_out.append(nods_pred.assign(overlap_index=lambda df: [np.nan] * nods_pred.shape[0]))
            continue
        try:
            nods_pred = pred_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX']]
        except KeyError:
            nods_true = true_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX']]
            true_out.append(nods_true.assign(overlap_index=lambda df: [np.nan] * nods_pred.shape[0]))
            continue

        overlap_matrix = nodules_sets_overlap_jit(nods_true.values, nods_pred.values)

        ov_mask_true, ov_ind_true = _create_overlap_index(overlap_matrix)
        ov_mask_pred, ov_ind_pred = _create_overlap_index(overlap_matrix.T)

        nods_true = nods_true.assign(overlap_index=lambda df: df.index)
        nods_true.loc[ov_mask_true, 'overlap_index'] = nods_pred.index[ov_ind_true[ov_mask_true]]
        nods_true.loc[np.logical_not(ov_mask_true), 'overlap_index'] = np.nan

        nods_pred = nods_pred.assign(overlap_index=lambda df: df.index)
        nods_pred.loc[ov_mask_pred, 'overlap_index'] = nods_true.index[ov_ind_pred[ov_mask_pred]]
        nods_pred.loc[np.logical_not(ov_mask_pred), 'overlap_index'] = np.nan

        true_out.append(nods_true)
        pred_out.append(nods_pred)

    ppl_nodules_true.append(pd.concat(true_out))
    ppl_nodules_pred.append(pd.concat(pred_out))
    return batch
