# pylint: disable=no-method-argument
"""Child class of CTImagesBatch that incorporates nn-models """
import sys
import functools
import tqdm
import numpy as np
import pandas as pd
from numba import njit
from IPython.display import clear_output
from ..preprocessing import CTImagesMaskedBatch
from ..dataset import action, Batch
from ..dataset import model as batch_model

from .utils import nodules_sets_overlap_jit


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


class CTImagesModels(CTImagesMaskedBatch):
    """ Сlass for describing, training nn-models of segmentation/classification;
            inference using models is also supported.
    """


    def unpack_component(self, component, dim_ordering):
        """ Basic way for unpacking 'images' or 'masks' from batch.

        Args:
        - component: str, component to unpack, can be 'images' or 'masks';
        in other case AttributeError will be raised;
        - dim_ordering: str, can be 'channels_last' or 'channels_first';
        reflects where to put channels dimension: right after batch
        dimension or after all spatial axes;

        Returns:
        - unpacked 'images' or 'masks' component as numpy array of the
        following shape: [BatchSize, 1, zdim, ydim, xdim]
        or [BatchSize, zdim, ydim, xdim, 1];
        """
        if component not in ('masks', 'images'):
            raise AttributeError("Component must be 'images' or 'masks'")
        x = np.stack([self.get(i, component) for i in range(len(self))])
        if dim_ordering == 'channels_last':
            x = x[..., np.newaxis]
        elif dim_ordering == 'channels_first':
            x = x[:, np.newaxis, ...]
        return x

    def unpack_seg(self, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for segmentation task.

        Args:
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - tuple(images_array, masks_array) with '1' in channels dimension;

        XXX 'dim_ordering' argument reflects where to put '1'
        for channels dimension both for images and masks.
        """
        return (self.unpack_component('images', dim_ordering),
                self.unpack_component('masks', dim_ordering))

    def unpack_clf(self, threshold=10, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for classification task.

        Args:
        - threshold: int, minimum number of '1' pixels in mask to consider it
        cancerous;
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - tuple(images_array, labels_array)

        XXX 'dim_ordering' argument reflects where to put '1' for channels dimension.
        """
        y_pred = np.asarray([self.get(i, 'masks').sum() > threshold
                             for i in range(len(self))], dtype=np.int)

        return self.unpack_component('images', dim_ordering), y_pred

    def unpack_reg(self, threshold=10, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for regression task.

        Args:
        - theshold: int, minimum number of '1' pixels in mask to consider it
        cancerous;
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - tuple(images_array, regression_array);

        XXX 'dim_ordering' argument reflects where to put '1' for channels dimension.
        """

        nods = self.nodules
        nodule_sizes = (nods.nodule_size / self.spacing[nods.patient_pos, :]
                        / self.images_shape[nods.patient_pos, :])

        nodule_centers = (nods.nodule_center / self.spacing[nods.patient_pos, :]
                          / self.images_shape[nods.patient_pos, :])

        sizes = np.zeros(shape=(len(self), 3), dtype=np.float)
        centers = np.zeros(shape=(len(self), 3), dtype=np.float)

        sizes[self.nodules.patient_pos, :] = nodule_sizes
        centers[self.nodules.patient_pos, :] = nodule_centers

        labels = np.expand_dims(self.unpack_clf(threshold, dim_ordering), axis=1)
        y_true = np.concatenate([centers, sizes, labels], axis=1)

        return self.unpack_images(dim_ordering), y_true

    def _get_by_unpacker(self, unpacker, **kwargs):
        """ Check unpacker type and get result from it. """
        if isinstance(unpacker, str):
            _unpacker = getattr(self, unpacker)
            if callable(_unpacker):
                x, y_true = _unpacker(**kwargs)
            else:
                x, y_true = _unpacker
        elif callable(unpacker):
            x, y_true = _unpacker(**kwargs)
        else:
            raise ValueError("Argument 'unpacker' can be name of batch instance"
                             + " attribute or external function. Got %s" % unpacker)
        return x, y_true

    @action
    def train_model(self, model_name, unpacker, **kwargs):
        """ Train model on crops of CT-scans contained in batch.

        Args:
        - model_name: str, name of classification model;
        - unpacker: callable or str, if str must be attribute of batch instance;
        if callable then it is called with '**kwargs' and its output is considered
        as data flowed in model.train_on_batch method;
        - component: str, name of y component, can be 'masks' or 'labels'(optional);
        - dim_ordering: str, dimension ordering, can be 'channels_first'
        or 'channels_last'(optional);

        Returns:
        - self, unchanged CTImagesMaskedBatch;
        """
        if self.pipeline is None:
            return self

        if self.pipeline.config is not None:
            train_iter = self.pipeline.get_variable('iter', 0)

            metrics = self.pipeline.config.get('metrics', ())
            show_metrics = self.pipeline.config.get('show_metrics', False)

            df_init = lambda: pd.DataFrame(columns=[m.__name__ for m in metrics])
            train_metrics = self.pipeline.get_variable('train_metrics', init=df_init)

        _model = self.get_model_by_name(model_name)
        x, y_true = self._get_by_unpacker(unpacker, **kwargs)
        _model.train_on_batch(x, y_true)

        if len(metrics):
            y_pred = _model_train.predict_on_batch(x)
            extend_data = {m.__name__: m(y_true, y_pred) for m in metrics}

            n = train_metrics.shape[0]
            train_metrics.loc[n, list(extend_data.keys())] = list(extend_data.values())

        if show_metrics:
            sys.stdout.write(str(train_metrics.iloc[-1, :]))
            clear_output(wait=True)

        self.pipeline.set_variable('iter', train_iter + 1)
        return self

    @action
    def predict_model(self, model_name, unpacker, **kwargs):
        """ Predict by model on crops of CT-scans contained in batch.

        Args:
        - model_name: str, name of classification model;
        - unpacker: callable or str, if str must be attribute of batch instance;
        if callable then it is called with '**kwargs' and its output is considered
        as data flowed in model.train_on_batch method;
        - component: str, name of y component, can be 'masks' or 'labels'(optional);
        - dim_ordering: str, dimension ordering, can be 'channels_first'
        or 'channels_last'(optional);

        Returns:
        - self, unchanged CTImagesMaskedBatch;
        TODO this method does not save result anywhere: in this version
        it's just useless;
        """
        _model = self.get_model_by_name(model_name)
        x, _ = self._get_by_unpacker(unpacker, **kwargs)
        _model.predict_on_batch(x)
        return self

    @action
    def test_on_dataset(self, model_name, unpacker, batch_size, period, **kwargs):
        """ Compute metrics of model on dataset.

        Args:
        - model_name: str, name of model;
        - unpacker: callable or str, if str must be attribute of batch instance;
        if callable then it is called with '**kwargs' and its output is considered
        as data flowed in model.predict_on_batch method;
        - component: str, name of y_component, can be 'masks' or 'labels'(optional);
        - dim_ordering: str, dimension ordering, can be 'channels_first'
        or 'channels last'(optional);
        - period: int, frequency of test_on_dataset runs;

        Returns:
        - self, unchanged CTImagesMaskedBatch;
        """
        if self.pipeline is None:
            return self

        if self.pipeline.config is not None:
            train_iter = self.pipeline.get_variable('iter', 0)

            metrics = self.pipeline.config.get('metrics', ())
            test_pipeline = self.pipeline.config.get('test_pipeline', None)
            test_pipeline.reset_iter()

            df_init = lambda: pd.DataFrame(columns=[m.__name__ for m in metrics])
            test_metrics = self.pipeline.get_variable('test_metrics', init=df_init)

        if len(metrics) and (train_iter % period == 0):
            _model = self.get_model_by_name(model_name)
            for batch in test_pipeline.gen_batch(batch_size):
                x, y_true = batch._get_by_unpacker(unpacker, **kwargs)

                y_pred = _model.predict_on_batch(x)

                extend_data = {m.__name__: m(y_true, y_pred) for m in metrics}

                n = test_metrics.shape[0]
                test_metrics.loc[n, list(extend_data.keys())] = list(extend_data.values())
        return self

    @action
    def predict_on_scan(self, model_name, strides=(16, 32, 32), crop_shape=(32, 64, 64),
                        batch_size=4, y_component='labels', dim_ordering='channels_last',
                        show_progress=True):
        """ Get predictions of the model on data contained in batch.

        Transforms scan data into patches of shape CROP_SHAPE and then feed
        this patches sequentially into model with name specified by
        argument 'model_name'; after that loads predicted masks or probabilities
        into 'masks' component of the current batch and returns it.

        Args:
        - model_name: str, name of model;
        - strides: tuple(int, int, int) strides for patching operation;
        - batch_size: int, number of patches to feed in model in one iteration;
        - y_component: str, name of y component, can be 'masks' or labels;
        - dim_ordering: str, dimension ordering, can be 'channels_first' or 'channels_last'

        Returns:
        - self, uncahnged CTImagesMaskedBatch;
        """
        _model = self.get_model_by_name(model_name)

        patches_arr = self.get_patches(patch_shape=crop_shape, stride=strides, padding='reflect')
        if dim_ordering == 'channels_first':
            patches_arr = patches_arr[:, np.newaxis, ...]
        elif dim_ordering == 'channels_last':
            patches_arr = patches_arr[..., np.newaxis]

        predictions = []
        iterations = range(0, patches_arr.shape[0], batch_size)
        if show_progress:
            iterations = tqdm.tqdm_notebook(iterations)  # pylint: disable=redefined-variable-type
        for i in iterations:
            current_prediction = np.asarray(_model.predict_on_batch(patches_arr[i: i + batch_size, ...]))

            if y_component == 'labels':
                current_prediction = np.stack([np.ones(shape=(crop_shape)) * prob
                                               for prob in current_prediction.ravel()])

            if y_component == 'regression':
                masks_patch = create_mask_reg(current_prediction[:, :3], current_prediction[:, 3:6],
                                              current_prediction[:, 6], crop_shape, 0.01)

                current_prediction = np.squeeze(masks_patch)
            predictions.append(current_prediction)

        patches_mask = np.concatenate(predictions, axis=0)
        self.load_from_patches(patches_mask, stride=strides,
                               scan_shape=tuple(self.images_shape[0, :]),
                               data_attr='masks')
        return self

    def _create_overlap_index(self, overlap_matrix):
        """ Get indices of nodules that overlaps using overlap_matrix. """
        argmax_ov = overlap_matrix.argmax(axis=1)
        max_ov = overlap_matrix.max(axis=1).astype(np.bool)

        return max_ov, argmax_ov

    @action
    def overlap_nodules(self):
        ppl_nodules_true = self.pipeline.get_variable('nodules_true', init=list)
        ppl_nodules_pred = self.pipeline.get_variable('nodules_pred', init=list)

        batch_nodules_true = self.nodules

        self.fetch_nodules_from_mask()
        batch_nodules_pred = self.nodules

        true_df = (
                      self.nodules_to_df(batch_nodules_true)
                          .set_index('nodule_id')
                          .assign(diam=lambda df: np.max(df.iloc[:, [4, 5, 6]], axis=1))
                  )

        pred_df = (
                      self.nodules_to_df(batch_nodules_pred)
                          .set_index('nodule_id')
                          .assign(diam=lambda df: np.max(df.iloc[:, [4, 5, 6]], axis=1))
                  )

        true_out = []
        pred_out = []
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
                true_out.append(nods_true.assign(overlap_index=lambda df: [np.nan] * nods_true.shape[0]))
                continue

            overlap_matrix = nodules_sets_overlap_jit(nods_true.values, nods_pred.values)

            ov_mask_true, ov_ind_true = self._create_overlap_index(overlap_matrix)
            ov_mask_pred, ov_ind_pred = self._create_overlap_index(overlap_matrix.T)

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
        return self
