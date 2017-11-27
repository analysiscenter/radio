# pylint: disable=too-many-arguments
# pylint: disable=not-context-manager
""" Contains DenseNet model class. """

import tensorflow as tf
from ....dataset.models.tf import DenseNet


class DenseNoduleNet(DenseNet):
    """ Implementation of custom DenseNet architecture for lung cancer detection.

    Full description of similar 2D model architecture can be downloaded from here:
    https://arxiv.org/pdf/1608.06993v2.pdf
    """
    @classmethod
    def default_config(cls):
        """ Sepcification of custom block parameters. """
        config = DenseNet.default_config()
        input_config = dict(layout='cnap', filters=16, kernel_size=7,
                            pool_size=3, pool_strides=(1, 2, 2))

        config['head'].update(dict(layout='Vdfna', activation=tf.nn.sigmoid))
        config['input_block'].update(input_config)
        config['body']['num_blocks'] = [6, 12, 24, 16]
        return config
