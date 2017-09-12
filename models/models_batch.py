# pylint: disable=no-method-argument
"""Child class of CTImagesBatch that incorporates nn-models """

import tensorflow as tf

from ..preprocessing import CTImagesMaskedBatch
from ..dataset import model

from .keras_unet import KerasUnet
from .keras_unet import dice_coef_loss, dice_coef, jaccard_coef, tiversky_loss
from .keras_model import KerasModel
# global constants
# input shape of a nodule

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
    def keras_unet(pipeline):
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
