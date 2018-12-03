# pylint: disable=too-few-public-methods
""" Contains DenseNoduleNet model class. """

from ...batchflow.models.tf import DenseNet         # pylint: disable=no-name-in-module


class DenseNoduleNet(DenseNet):
    """ Implementation of custom DenseNet architecture for lung cancer detection. """
    @classmethod
    def default_config(cls):
        """ Specification of custom block parameters. """
        config = DenseNet.default_config()
        input_config = dict(layout='cnap', filters=16, kernel_size=7,
                            pool_size=3, pool_strides=(1, 2, 2))

        config['initial_block'].update(input_config)      # pylint: disable=no-member
        config['body']['num_blocks'] = [6, 12, 24, 16]  # pylint: disable=invalid-sequence-index
        return config
