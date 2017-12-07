""" Iandola F. et al. "`SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
<https://arxiv.org/abs/1602.07360>`_"
"""
import numpy as np
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class SqueezeNet(TFModel):
    """ SqueezeNet neural network

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)

    body : dict
        layout : str
            A sequence of blocks:

            - f : fire block
            - m : max-pooling
            - b : bypass

            Default is 'fffmffffmf'.
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['input_block'].update(dict(layout='cnap', filters=96, kernel_size=7, strides=2,
                                          pool_size=3, pool_strides=2))
        config['body']['layout'] = 'fffmffffmf'
        #config['body']['layout'] = 'ffbfmbffbffmbf'

        num_blocks = len(config['body']['layout'])
        layers_filters = 16 * 2 ** np.arange(num_blocks//2)
        layers_filters = np.repeat(layers_filters, 2)[:num_blocks].copy()
        config['body']['filters'] = layers_filters

        config['head'].update(dict(layout='dcnaV', kernel_size=1, strides=1, dropout_rate=.5))

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['filters'] = self.num_classes('targets')

        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create base VGG layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of block types
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        layout = kwargs.pop('layout')
        filters = kwargs.pop('filters')

        x = inputs
        bypass = None
        with tf.variable_scope(name):
            for i, block in enumerate(layout):
                if block == 'b':
                    bypass = x
                if block == 'f':
                    x = cls.fire_block(x, filters=filters[i], name='fire-block-%d' % i, **kwargs)
                elif block == 'm':
                    x = conv_block(x, 'p', name='max-pool-%d' % i, **kwargs)

                if bypass is not None:
                    bypass_channels = cls.channels_shape(bypass, kwargs.get('data_format'))
                    x_channels = cls.channels_shape(x, kwargs.get('data_format'))

                    if x_channels != bypass_channels:
                        bypass = conv_block(bypass, 'c', x_channels, 1, name='bypass-%d' % i, **kwargs)
                    x = x + bypass
                    bypass = None
        return x

    @classmethod
    def fire_block(cls, inputs, filters, layout='cna', name='fire-block', **kwargs):
        """ A sequence of 3x3 and 1x1 convolutions followed by pooling

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            the number of filters in each convolution layer

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = conv_block(inputs, layout, filters, 1, name='squeeze-1x1', **kwargs)

            exp1 = conv_block(x, layout, filters*4, 1, name='expand-1x1', **kwargs)
            exp3 = conv_block(x, layout, filters*4, 3, name='expand-3x3', **kwargs)

            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat([exp1, exp3], axis=axis)
        return x
