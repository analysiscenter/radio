from .models import KerasUnet
from .models.keras_unet import dice_coef_loss, dice_coef, jaccard_coef, tiversky_loss
from ..dataset import action, model

class BatchModels(object):

    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)

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
        config = pipeline.config
        if config.get("path", None) is not None:
            unet = KerasModel('unet')

            if config.get('loss') == 'tiversky':
                custom_objects = {'tiversky_loss': tiversky_loss,
                                  'dice_coef': dice_coef,
                                  'jaccard_coef': jaccard_coef,
                                  'dice_coef_loss': dice_coef_loss}

            elif config.get('loss') == 'dice':
                custom_objects = {'dice_coef_loss': dice_coef_loss,
                                  'dice_coef': dice_coef}
        else:
            unet = KerasUnet('unet')
            loss_dict = {'dice': dice_loss, 'tiversky_loss': tiversky_loss}
            unet.compile(optimizer='adam', loss=loss_dict[config.get('loss', 'dice')])
        return unet

    @model(mode='static')
    def keras_vgg16(pipeline):
        """ Create VGG16 model implemented in keras and immediatelly compile it.

        This method is wrapped with model(mode='static' decorator meaning
        that model is created in the momment when pipeline.run(...) is called.
        Config attribute is a dictionary that can contain 'path' and '' values.

        Args:
        - pipeline: Pipeline object from dataset package; it is the only argument
        and it is used to pass parameters required by model being created
        through pipeline.config attribute;

        Returns:
        - compiled model;
        """
        return None

    @model(mode='static')
    def keras_resnet50(pipeline):
        """ Create ResNet50 model implemented in keras and immediatelly compile it.

        This method is wrapped with model(mode='static') decorator meaning
        that model is created in the momment when pipeline.run(...) is called.
        Config attribute is a dictionary that can contain 'path' and '' values.

        Args:
        - pipeline: Pipeline object from dataset package; it is the only argument
        and it is used to pass parameters required by model through pipeline.config
        attribute.

        Returns:
        - compiled model.
        """
        pass
