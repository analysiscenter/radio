# pylint: disable=too-few-public-methods
""" Contains DenseNoduleNet model class. """

import tensorflow as tf
from ....dataset.dataset.models.tf import DenseNet


class DenseNoduleNet(DenseNet):
    """ Implementation of custom DenseNet architecture for lung cancer detection. """
    @classmethod
    def default_config(cls):
        """ Sepcification of custom block parameters. """
        config = DenseNet.default_config()
        input_config = dict(layout='cnap', filters=16, kernel_size=7,
                            pool_size=3, pool_strides=(1, 2, 2))

        config['input_block'].update(input_config)
        config['body']['num_blocks'] = [6, 12, 24, 16]
        return config
