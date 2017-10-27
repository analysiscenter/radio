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

from .utils import nodules_sets_overlap_jit, create_mask_reg


class CTImagesModels(CTImagesMaskedBatch):
    """ Contains methods for transforming batch instance to dictionaries
    that can be fed into models.

    Unpack methods:
    - unpack_component: unpack batch component(component can be 'masks' or 'images');

    - unpack_seg: unpack batch into dictionary suitable for segmentation neural networks;
      Ouput dictionary looks like:
      {'x': ndarray(batch_size, size_x, size_y, size_z, 1),
       'y': ndarray(batch_size, size_x, size_y, size_z, 1)}

      'x' contains batch of source crops, 'y' contains batch of corresponding masks;

    - unpack_reg: unpack batch into dictionary suitable for regression neural networks;
      Output dictionary looks like:
      {'x': ndarray(batch_size, size_x, size_y, size_z, 1),
       'y': ndarray(batch_size, 7)}

       'x' contains batch of source crops, 'y' contains batch of 7-dim vectors
       with probabilities of cancer, sizes and centers;

    - unpack_clf: unpack batch into dictionary suitable for classification neural networks;
      Output dictionary looks like:
      {'x': ndarray(batch_size, size_x, size_y, size_z, 1),
       'y': ndarray(batch_size, 1)}

       'x' contains batch of source crops, 'y' contains batch of 1-dim vectors
       with 0 or 1;

    """

    def unpack_component(batch, model, component, dim_ordering):
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
        x = np.stack([batch.get(i, component) for i in range(len(batch))])
        if dim_ordering == 'channels_last':
            x = x[..., np.newaxis]
        elif dim_ordering == 'channels_first':
            x = x[:, np.newaxis, ...]
        return x

    def unpack_seg(batch, model, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for segmentation task.

        Args:
        - self: CTImagesModels batch instance;
        - model: instance of dataset.models.BaseModel, is not used here;
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - dict {'x': images_array, 'y': labels_array}

        XXX 'dim_ordering' argument reflects where to put '1'
        for channels dimension both for images and masks.
        """
        return {'feed_dict': {'x': batch.unpack_component(model, 'images', dim_ordering),
                              'y': batch.unpack_component(model, 'masks', dim_ordering)}}

    def unpack_clf(batch, model, threshold=10, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for classification task.

        Args:
        - self: CTImagesModels batch instance;
        - model: instance of dataset.models.BaseModel, is not used here;
        - threshold: int, minimum number of '1' pixels in mask to consider it
        cancerous;
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - dict {'x': images_array, 'y': labels_array}

        XXX 'dim_ordering' argument reflects where to put '1' for channels dimension.
        """
        masks_labels = np.asarray([batch.get(i, 'masks').sum() > threshold
                                   for i in range(len(batch))], dtype=np.int)

        return {'feed_dict': {'x': batch.unpack_component(model, 'images', dim_ordering),
                              'y': masks_labels[:, np.newaxis]}}

    def unpack_reg(batch, model, threshold=10, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for regression task.

        Args:
        - self: CTImagesModels batch instance;
        - model: instance of dataset.models.BaseModel, is not used here;
        - theshold: int, minimum number of '1' pixels in mask to consider it
        cancerous;
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - dict {'x': images_array, 'y': y_regression_array}

        TODO Test this method;
        XXX 'dim_ordering' argument reflects where to put '1' for channels dimension.
        """

        nods = batch.nodules

        sizes = np.zeros(shape=(len(batch), 3), dtype=np.float)
        centers = np.zeros(shape=(len(batch), 3), dtype=np.float)

        for patient_pos, _ in enumerate(batch.indices):
            pat_nodules = nods[nods.patient_pos == patient_pos]
            mask_nod_indices = pat_nodules.nodule_size.max(axis=1).argmax()

            if len(pat_nodules) == 0:
                continue

            nodule_sizes = (pat_nodules.nodule_size / batch.spacing[patient_pos, :]
                            / batch.images_shape[patient_pos, :])

            nodule_centers = (pat_nodules.nodule_center / batch.spacing[patient_pos, :]
                              / batch.images_shape[patient_pos, :])

            sizes[patient_pos, :] = nodule_sizes[mask_nod_indices, :]
            centers[patient_pos, :] = nodule_centers[mask_nod_indices, :]

        x, labels = batch.unpack_clf(threshold, dim_ordering)
        labels = np.expand_dims(labels, axis=1)
        y_regression_array = np.concatenate([centers, sizes, labels], axis=1)

        return {'feed_dict': {'x': x, 'y': y_regression_array}}


    # @action
    # def train_model(self, model_name, unpacker, **kwargs):
    #     """ Train model on crops of CT-scans contained in batch.
    #
    #     Args:
    #     - model_name: str, name of classification model;
    #     - unpacker: callable or str, if str must be attribute of batch instance;
    #     if callable then it is called with '**kwargs' and its output is considered
    #     as data flowed in model.train_on_batch method;
    #     - component: str, name of y component, can be 'masks' or 'labels'(optional);
    #     - dim_ordering: str, dimension ordering, can be 'channels_first'
    #     or 'channels_last'(optional);
    #
    #     Returns:
    #     - self, unchanged CTImagesMaskedBatch;
    #     """
    #     if self.pipeline is None:
    #         return self
    #
    #     if self.pipeline.config is not None:
    #         train_iter = self.pipeline.get_variable('iter', 0)
    #
    #         metrics = self.pipeline.config.get('metrics', ())
    #         show_metrics = self.pipeline.config.get('show_metrics', False)
    #         train_metrics = self.pipeline.get_variable('train_metrics', init=list)
    #
    #     _model = self.get_model_by_name(model_name)
    #     x, y_true = self._get_by_unpacker(unpacker, **kwargs)
    #     _model.train_on_batch(x, y_true)
    #
    #     if len(metrics):
    #         y_pred = _model.predict_on_batch(x)
    #         extend_data = {m.__name__: m(y_true, y_pred) for m in metrics}
    #         train_metrics.append(extend_data)
    #
    #     if show_metrics:
    #         sys.stdout.write(str(pd.Series(train_metrics[-1])))
    #         clear_output(wait=True)
    #
    #     self.pipeline.set_variable('iter', train_iter + 1)
    #     return self

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

            test_metrics = self.pipeline.get_variable('test_metrics', init=list)

        if len(metrics) and (train_iter % period == 0):
            _model = self.get_model_by_name(model_name)
            ds_metrics_list = []
            for batch in test_pipeline.gen_batch(batch_size):
                x, y_true = batch._get_by_unpacker(unpacker, **kwargs)

                y_pred = _model.predict_on_batch(x)

                extend_data = {m.__name__: m(y_true, y_pred) for m in metrics}

                ds_metrics_list.append(extend_data)

            ds_metrics = pd.DataFrame(ds_metrics_list).mean()
            test_metrics.append(ds_metrics.to_dict())
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
        """ Accumulate info about true and predicted nodules in pipeline variables. """
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
