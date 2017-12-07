"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """ UNet

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[128, 256, 512, 1024])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        filters = 64   # number of filters in the first block

        config['input_block'].update(dict(layout='cna cna', filters=filters, kernel_size=3, strides=1))
        config['body']['upsampling_kernel'] = 2
        config['body']['num_blocks'] = 4
        config['body']['filters'] = 2 ** np.arange(config['body']['num_blocks']) * filters * 2
        config['head'].update(dict(layout='cna cna', filters=filters, kernel_size=3, strides=1))
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in downsampling blocks
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
            encoder_outputs = [x]
            for i, ifilters in enumerate(filters):
                x = cls.downsampling_block(x, ifilters, name='downsampling-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[::-1]):
                x = cls.upsampling_block((x, encoder_outputs[-i-2]), ifilters//2, name='upsampling-'+str(i), **kwargs)

        return x

    @classmethod
    def downsampling_block(cls, inputs, filters, name, **kwargs):
        """ 2x2 max pooling with stride 2 and two 3x3 convolutions

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
        x = conv_block(inputs, 'pcnacna', filters, kernel_size=3, name=name, pool_size=2, pool_strides=2, **kwargs)
        return x

    @classmethod
    def upsampling_block(cls, inputs, filters, name, **kwargs):
        """ 3x3 convolution and 2x2 transposed convolution

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
        config = cls.fill_params('body', **kwargs)
        kernel = cls.pop('upsampling_kernel', config)
        with tf.variable_scope(name):
            x, skip = inputs
            x = conv_block(x, 'tna', filters, kernel, name='upsample', strides=2, **kwargs)
            x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat((skip, x), axis=axis)
            x = conv_block(x, 'cnacna', filters, kernel_size=3, name='conv', **kwargs)
        return x

    @classmethod
    def head(cls, inputs, num_classes, name='head', **kwargs):
        """ Conv block followed by 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        with tf.variable_scope(name):
            x = conv_block(inputs, name='conv', **kwargs)
            x = conv_block(inputs, name='last', **{**kwargs, **dict(filters=num_classes, kernel_size=1, layout='c')})
        return x
