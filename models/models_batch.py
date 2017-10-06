# pylint: disable=no-method-argument
"""Child class of CTImagesBatch that incorporates nn-models """
import os
import sys
import tqdm
import numpy as np
import pandas as pd
from numba import njit
from IPython.display import clear_output
from ..preprocessing import CTImagesMaskedBatch
from ..dataset import model, action

from .keras.keras_unet import KerasUnet
from .keras.losses import dice_coef_loss, dice_coef, jaccard_coef, tiversky_loss
from .keras.architectures.keras_model import KerasModel


@njit(nogil=True)
def create_mask_jit(masks, start, end):
    num_items = start.shape[0]
    for i in range(num_items):
        masks[i,
              start[i, 0]: end[i, 0],
              start[i, 1]: end[i, 1],
              start[i, 2]: end[i, 2]] = 1.
    return masks


def create_mask_reg(centers, sizes, probs, crop_shape, threshold):
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

    Methods:
        1. selu_vnet_4:
            build vnet of depth = 4 using tensorflow,
            return tensors necessary for training, evaluating and inferencing
        2. train_vnet_4:
            train selu_vnet_4 on images and masks, that are contained in batch
        3. update_test_stats:
            method for evaluation of the model on test-batch (test-dataset) during
            training
        4. get_cancer_segmentation:
            method for performing inference on images of batch using trained model
    """

    @model(mode='static')
    def keras_unet(pipeline):  # pylint: disable=no-self-argument
        """ Create Unet model implemented in keras and immediatelly compile it.

        This method is wrapped with model(mode='static') decorator meaning
        that model is created in the momment when pipeline.run(...) is called.
        Config attribute is a dictionary that can contain 'loss' and 'path'
        values. If config contains 'path', then model is loaded from directory
        specified by this parameter. Otherwise new keras model is to be built;

        Another key of config dict is 'loss'. Value 'loss' can be one of
        two str: 'dice' or 'tiversky'. This value specifies the loss function
        that will be used during model compilation;

        Args:
        - pipeline: Pipeline object from dataset package; it is the only argument
        and it is used to pass parameters required by model being created
        through pipeline.config attribute;

        Returns:
        - compiled model;
        """
        path = pipeline.config.get("path", None)
        loss = pipeline.config.get("loss", 'dice')
        if  path is not None:
            unet = KerasModel('unet')

            if loss == 'tiversky':
                custom_objects = {'tiversky_loss': tiversky_loss,
                                  'dice_coef': dice_coef,
                                  'jaccard_coef': jaccard_coef,
                                  'dice_coef_loss': dice_coef_loss}

            elif loss == 'dice':
                custom_objects = {'dice_coef_loss': dice_coef_loss,
                                  'dice_coef': dice_coef}

            unet.load_model(path, custom_objects=custom_objects)
        else:
            unet = KerasUnet('unet')  # pylint: disable=redefined-variable-type
            loss_dict = {'dice': dice_coef_loss, 'tiversky_loss': tiversky_loss}
            unet.compile(optimizer='adam', loss=loss_dict[loss])
        return unet

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
        y_pred =  np.asarray([self.get(i, 'masks').sum() > threshold
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
        metrics = ()
        show_metrics = False
        if self.pipeline.config is not None:
            metrics = self.pipeline.config.get('metrics', ())
            show_metrics = self.pipeline.config.get('show_metrics', False)

            df_init = lambda: pd.DataFrame(columns=[m.__name__ for m in metrics])
            train_metrics = self.pipeline.get_variable('train_metrics', init=df_init)

        _model = self.get_model_by_name(model_name)
        x, y_true = self._get_by_unpacker(unpacker, **kwargs)
        _model.train_on_batch(x, y_true)

        if len(metrics):
            y_pred = _model.predict_on_batch(x)
            train_metrics.append({m.__name__: m(y_true, y_pred) for m in metrics})

        if show_metrics:
            sys.stdout.write(str(train_metrics.iloc[-1, :]))
            clear_output(wait=True)
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
        """
        _model = self.get_model_by_name(model_name)
        x, _ = self._get_by_unpacker(unpacker, **kwargs)
        _model.predict_on_batch(x)
        return self

    @action
    def test_on_dataset(self, model_name, unpacker, batch_size, **kwargs):
        """ Compute metrics of model on dataset.

        Args:
        - model_name: str, name of model;
        - unpacker: callable or str, if str must be attribute of batch instance;
        if callable then it is called with '**kwargs' and its output is considered
        as data flowed in model.predict_on_batch method;
        - component: str, name of y_component, can be 'masks' or 'labels'(optional);
        - dim_ordering: str, dimension ordering, can be 'channels_first'
        or 'channels last'(optional);

        Returns:
        - self, unchanged CTImagesMaskedBatch;
        """
        metrics = ()
        show_metrics = False
        if pipeline.config is not None:
            metrics = self.pipeline.config.get('metrics', ())
            test_pipeline = self.pipeline.config.get('test_pipeline', None)
            test_dataset = self.pipeline.config.get('test_dataset', None)

            df_init = lambda: pd.DataFrame(columns=[m.__name__ for m in metrics])
            test_metrics = self.pipeline.get_variable('test_metrics', init=df_init)

        if len(metrics):
            _model = self.get_model_by_name(model_name)
            x, _ = self._get_by_unpacker(unpacker, **kwargs)

            y_pred = _model.predict_on_batch(x)

            for batch in (test_pipeline << test_dataset).gen_batch(batch_size):
                test_metrics.append({m.__name__: m(y_true, y_pred) for m in metrics})
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
            iterations = tqdm.tqdm_notebook(iterations)
        for i in iterations:
            current_prediction = np.asarray(_model.predict_on_batch(patches_arr[i: i + batch_size, ...]))

            if y_component == 'labels':
                current_prediction = np.stack([np.ones(shape=(crop_shape)) * prob
                                               for prob in current_prediction.ravel()])

            if y_component == 'regression':
                masks_patch = create_mask_reg(current_prediction[:, :3], current_prediction[:, 3:6],
                                              current_prediction[:, 6], crop_shape, 0.01)

                current_prediction = np.squeeze(current_prediction)
            predictions.append(current_prediction)

        patches_mask = np.concatenate(predictions, axis=0)
        self.load_from_patches(patches_mask, stride=strides,
                               scan_shape=tuple(self.images_shape[0, :]),
                               data_attr='masks')
        return self

    @action
    def binarize_mask(self, threshold=0.35):
        """ Binarize masks by threshold.

        Args:
        - threshold: float, threshold for masks binarization;

        Returns:
        - self, source CTImagesMaskedBatch with binarized masks.
        """
        self.masks *= np.asarray(self.masks > threshold, dtype=np.int)
        return self
