""" Different useful functions when working with models and CTImagesMaskedBatch. """

import numpy as np
import pandas as pd
from numba import njit

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = lambda *args, **kwargs: None


class MetricsAccumulator(object):  # pylint: disable=too-many-instance-attributes
    """ Metrics accumulator class for storing metrics history during learning.

    Enables creation of callables objects that store history of metrics
    over processed batches.

    Example
    -------

    Calling from train pipeline loop:

    >>> compute_metrics = MetricsAccumulator('resnet', unpack_clf,
    ...                                      [accuracy, recall, precision],
    ...                                      test_ppl=None, show=True)
    >>> train_ppl = (
    ...     ds.Pipeline()
    ...       .load(...)
    ...       .init_model('static', ResNetModel, 'resnet', resnet_config)
    ...       .train_model('resnet', unpack_clf)
    ...       .call(compute_metrics)
    ... ) << train_dataset
    >>> train_ppl.run(batch_size=4)
    """

    def __init__(self, model_name, unpacker, metrics=(), test_ppl=None, show=True):
        """ Create metrics accumulator.

        The main purpose of this class is creation of instances that
        will make the process of computation accumulation of metrics values
        inside ds.Pipeline's train loop easier.

        Parameters
        ----------
        model_name : str
            name of model that will be used for metrics computation.
        unpacker : callable(batch, model) -> {'x': ndarray, 'y': ndarray}
            function that will be used for unpacking the data from batch.
        metrics : tuple or list of callables
            container with metrics functions. Metrics function must
            accept two arguments: y_true and y_pred ndarrays.
        test_ppl : ds.Pipeline or None
            pipeline for test dataset. If not None then will be used for
            test batch generation.
        show : bool
            whether pring metrics values on each iteration.

        Example
        -------

        Calling from train pipeline loop:

        >>> compute_metrics = MetricsAccumulator('resnet', unpack_clf,
        ...                                      [accuracy, recall, precision],
        ...                                      test_ppl=None, show=True)
        >>> train_ppl = (
        ...     ds.Pipeline()
        ...       .load(...)
        ...       .init_model('static', ResNetModel, 'resnet', resnet_config)
        ...       .train_model('resnet', unpack_clf)
        ...       .call(compute_metrics)
        ... ) << train_dataset
        >>> train_ppl.run(batch_size=4)

        Note
        ----
        If `test_ppl` parameter in `__init__` is specified then computation of
        metrics on each train batch will be coupled with computation of the same
        metrics on test batch generated from test pipeline specified by
        `test_ppl` argument.
        """
        self.model_name = model_name
        self.unpacker = unpacker
        self.metrics = metrics
        self.test_ppl = test_ppl
        self._show_cond = show

        self._test_gen = None if test_ppl is None else self.ppl_generator(test_ppl)
        self._train_values = []
        self._test_values = []
        self._iter = 0

    @staticmethod
    def ppl_generator(ppl, batch_size=None):
        """ Convert input pipeline into infinite generator.

        Parameters
        ----------
        ppl : ds.Pipeline
            pipeline for test dataset.
        batch_size : int or None
            number elements in batch generated from ppl. If None then
            batch size must be specified in lazy run.

        Yields
        ------
        Batch
            next batch from ppl.
        """
        _ppl = ppl.run(lazy=True, batch_size=batch_size) if batch_size else ppl
        while True:
            try:
                batch = _ppl.next_batch()
            except StopIteration:
                _ppl.reset_iter()
                batch = _ppl.next_batch()
            finally:
                yield batch

    @property
    def train_df(self):
        """ Get dataframe with train metrics.

        Returns
        -------
        pd.DataFrame
            dataframe with train metrics.
        """
        return pd.DataFrame(self._train_values)

    @property
    def test_df(self):
        """ Get dataframe with test metrics.

        Returns
        -------
        pd.DataFrame
            dataframe with test metrics.
        """
        return None if self._test_gen is None else pd.DataFrame(self._test_values)

    def refresh(self):
        """ Refresh accumulator object.

        Refresh train and test metrics lists. Set number of iteration to zero.
        """
        self._iter = 0
        self._train_values = []
        self._test_values = []

    def call(self, batch, model):
        """ Compute metrics on batch with given model.

        Parameters
        ----------
        batch : ds.Batch
            input batch, data from this batch will be used for metrics computation.
        model : ds.models.BaseModel
            model which `predict` method will be used for getting `y_pred` values.

        Returns
        -------
        dict
            dictionary with metrics values. Keys are names of metrics.
        """
        batch_data = self.unpacker(batch, None)
        x = batch_data['x']
        y_true = batch_data['y']
        y_pred = model.predict(x)
        return {m.__name__: m(y_true, y_pred) for m in self.metrics}

    def __call__(self, batch):
        """ Compute all metrics on input batch.

        Parameters
        ----------
        batch : ds.Batch
            input train batch.

        Example
        -------
        Applying on a single batch:

        >>> metrics_accumulator = MetricsAccumulator('resnet', unpack_clf,
        ...                                          [accuracy, recall, precision],
        ...                                          test_ppl=None, show=True)
        >>> train_batch = train_ppl.next_batch()
        >>> metrics_accumulator(train_batch)

        Calling from train pipeline loop:

        >>> compute_metrics = MetricsAccumulator('resnet', unpack_clf,
        ...                                      [accuracy, recall, precision],
        ...                                      test_ppl=None, show=True)
        >>> train_ppl = (
        ...     ds.Pipeline()
        ...       .load(...)
        ...       .init_model('static', ResNetModel, 'resnet', resnet_config)
        ...       .train_model('resnet', unpack_clf)
        ...       .call(compute_metrics)
        ... ) << train_dataset
        >>> train_ppl.run(batch_size=4)

        """
        model = batch.get_model_by_name(self.model_name)
        self._train_values.append(self.call(batch, model))
        if self._test_gen:
            test_batch = next(self._test_gen)
            self._test_values.append(self.call(test_batch, model))

        if self._show_cond:
            self.show()
        self._iter += 1

    def show(self):
        """ Print dictionary containing metrics.

        Prints values of metrics on train and test batches gatgered in last __call__.

        Note
        ----
        This function is called inside __call__ if
        MetricsAccumulator object was created with argument `show=True`.
        """
        print('-----------------------------------')
        print(pd.Series(self._train_values[-1], name='train'))
        if self._test_gen is not None:
            print('\n')
            print(pd.Series(self._test_values[-1], name='test'))
        print('-----------------------------------')
        print('iteration: ', self._iter)
        print('-----------------------------------')
        clear_output(wait=True)

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
    ndarray(batch_size, zdim, ydim, xdim, 1) or None
        unpacked 'images' or 'masks' component of batch as numpy array.
    """
    if component not in ('masks', 'images'):
        return None

    if np.all(batch.images_shape == batch.images_shape[0, :]):
        x = batch.get(None, component).reshape(-1, batch.images_shape[0, :])
    else:
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
    batch : CTImagesMaskedBatch
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

    Note
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
            pred_out.append(nods_pred.assign(overlap_index=lambda df: [np.nan] * nods_pred.shape[0]))  # pylint: disable=cell-var-from-loop
            continue
        try:
            nods_pred = pred_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX']]
        except KeyError:
            nods_true = true_gr.get_group(group_name).loc[:, ['diam', 'locZ', 'locY', 'locX']]
            true_out.append(nods_true.assign(overlap_index=lambda df: [np.nan] * nods_pred.shape[0]))  # pylint: disable=cell-var-from-loop
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
