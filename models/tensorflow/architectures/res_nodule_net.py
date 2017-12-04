# pylint: disable=too-few-public-methods
""" Contains ResNodule3DNet50 model class. """

import numpy as np
import tensorflow as tf
from ....dataset.dataset.models.tf import ResNet


class ResNodule3DNet50(ResNet):
    """ Implementation of custom DenseNet architecture for lung cancer detection. """

    @classmethod
    def default_config(cls):
        """ Sepcification of custom block parameters. """
        config = ResNet.default_config()

        input_config = dict(layout='cnap', filters=16, kernel_size=7,
                            pool_size=3, pool_strides=(1, 2, 2))
        config['input_block'].update(input_config)

        config['body']['num_blocks'] = [3, 4, 6, 3]
        filters = 16   # number of filters in the first block
        config['body']['filters'] = 2 ** np.arange(len(config['body']['num_blocks'])) * filters \
                                    * config['body']['block']['width_factor']

        config['head'].update(dict(layout='Vdfna', activation=tf.nn.sigmoid))
        config['body']['block']['bottleneck'] = True
        return config
