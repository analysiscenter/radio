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
        unpack_component : unpack batch component(component can be 'masks' or 'images').

        unpack_seg : unpack batch into dictionary suitable for segmentation neural networks;
                     Ouput dictionary looks like:
                     {'x': ndarray(batch_size, size_x, size_y, size_z, 1),
                     'y': ndarray(batch_size, size_x, size_y, size_z, 1)}

                     'x' contains batch of source crops, 'y' contains batch of corresponding masks.

        unpack_reg : unpack batch into dictionary suitable for regression neural networks.
                     Output dictionary looks like:
                     {'x': ndarray(batch_size, size_x, size_y, size_z, 1),
                      'y': ndarray(batch_size, 7)}

                     'x' contains batch of source crops, 'y' contains batch of 7-dim vectors
                     with probabilities of cancer, sizes and centers.

        unpack_clf : unpack batch into dictionary suitable for classification neural networks.
                     Output dictionary looks like:
                     {'x': ndarray(batch_size, size_x, size_y, size_z, 1),
                      'y': ndarray(batch_size, 1)}

                     'x' contains batch of source crops, 'y' contains batch of 1-dim vectors
                     with 0 or 1.
    """

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
        return {'x': batch.unpack_component(model, 'images', dim_ordering),
                'y': batch.unpack_component(model, 'masks', dim_ordering)}

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

        return {'x': batch.unpack_component(model, 'images', dim_ordering),
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

        clf_dict = batch.unpack_clf(model, threshold, dim_ordering)
        x, labels = clf_dict['x'], clf_dict['y']
        y_regression_array = np.concatenate([centers, sizes, labels], axis=1)

        return {'x': x, 'y': y_regression_array}

    @action
    def test_on_dataset(self, model_name, unpacker, **kwargs):
        """ Compute metrics of model on dataset.

        Parameters
        ----------
        model_name : str
            name of model.
        unpacker : callable or str
            if str must be attribute of batch instance.
            if callable then it is called with '**kwargs' and its output is considered
            as data flowed in model.predict_on_batch method.
        component : str
            name of y_component, can be 'masks' or 'labels'(optional).
        dim_ordering : str
            dimension ordering, can be 'channels_first' or 'channels last'(optional).
        period: int
            frequency of test_on_dataset runs.

        Returns
        -------
        CTImagesMaskedBatch
            unchanged self(source batch).
        """
        if self.pipeline is None:
            return self
        train_iter = self.pipeline.get_variable('iter', 0)
        period = 32
        if self.pipeline.config is not None:
            period = self.pipeline.config.get('period', 32)

        if (train_iter % period == 0):
            _model = self.get_model_by_name(model_name)
            _model.test_on_dataset(partial(unpacker, **kwargs))
        self.pipeline.set_variable('iter', train_iter + 1)
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

        Parameters
        ----------
        model_name : str
            name of model that will be used for predictions.
        strides : tuple(int, int, int)
            strides for patching operation
        batch_size : int
            number of patches to feed in model in one iteration.
        y_component: str
            name of y component, can be 'masks' or labels.
        dim_ordering: str
            dimension ordering, can be 'channels_first' or 'channels_last'.

        Returns
        -------
        CTImagesMaskedBatch
            unchanged self(source batch).
        """
        _model = self.get_model_by_name(model_name)

        patches_arr = self.get_patches(patch_shape=crop_shape,
                                       stride=strides,
                                       padding='reflect')
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
                masks_patch = create_mask_reg(current_prediction[:, :3],
                                              current_prediction[:, 3:6],
                                              current_prediction[:, 6],
                                              crop_shape, 0.01)

                current_prediction = np.squeeze(masks_patch)
            predictions.append(current_prediction)

        patches_mask = np.concatenate(predictions, axis=0)
        self.load_from_patches(patches_mask, stride=strides,
                               scan_shape=tuple(self.images_shape[0, :]),
                               data_attr='masks')
        return self
