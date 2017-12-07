""" Chaurasia A., Culurciello E. "`LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
<https://arxiv.org/abs/1707.03718>`_"
"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel
from .resnet import ResNet


class LinkNet(TFModel):
    """ LinkNet

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[64, 128, 256, 512])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        filters = 64   # number of filters in the first block

        config['input_block'].update(dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
                                          pool_size=3, pool_strides=2))
        config['body']['num_blocks'] = 4
        config['body']['filters'] = 2 ** np.arange(config['body']['num_blocks']) * filters

        config['head']['filters'] = filters // 2

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ LinkNet body

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters = kwargs.pop('filters')

        with tf.variable_scope(name):
            x = inputs
            encoder_outputs = []
            for i, ifilters in enumerate(filters):
                x = cls.encoder_block(x, filters=ifilters, name='encoder-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[::-1][1:]):
                x = cls.decoder_block(x, filters=ifilters, name='decoder-'+str(i), **kwargs)
                x = cls.crop(x, encoder_outputs[-i-2], data_format=kwargs.get('data_format'))
                x = tf.add(x, encoder_outputs[-2-i])
            x = cls.upsampling_block(x, filters[0], 'decoder-'+str(i+1), **kwargs)
            x = cls.crop(x, inputs, data_format=kwargs.get('data_format'))

        return x

    @classmethod
    def encoder_block(cls, inputs, filters, name, **kwargs):
        """ Two ResNet blocks of two 3x3 convolution + shortcut

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        return ResNet.double_block(inputs, filters=filters, name=name, downsample=True, **kwargs)

    @classmethod
    def decoder_block(cls, inputs, filters, name, **kwargs):
        """ 1x1 convolution, 3x3 transposed convolution with stride=2 and 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        num_filters = inputs.get_shape()[-1].value // 4
        return conv_block(inputs, 'cna tna cna', [num_filters, num_filters, filters], [1, 3, 1],
                          name=name, strides=[1, 2, 1], **kwargs)

    @classmethod
    def head(cls, inputs, num_classes, name='head', **kwargs):
        """ 3x3 transposed convolution, 3x3 convolution and 2x2 transposed convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes (and number of filters in the last convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        filters = kwargs.pop('filters')

        x = conv_block(inputs, 'tna cna t', [filters, filters, num_classes], [3, 3, 2],
                       strides=[2, 1, 2], name=name, **kwargs)
        return x
