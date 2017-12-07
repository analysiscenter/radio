""" Howard A. G. et al. "`MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
<https://arxiv.org/abs/1704.04861>`_"
"""

import tensorflow as tf

from . import TFModel
from .layers import conv_block


_DEFAULT_BODY_ARCH = {
    'strides': [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2],
    'double_filters': [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    'width_factor': 1
}


class MobileNet(TFModel):
    """ MobileNet

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`.TFModel._make_inputs`)

    input_block : dict

    body : dict
        strides : list of int
            strides in separable convolutions

        double_filters : list of bool
            if True, number of filters in 1x1 covolution will be doubled

        width_factor : float
            multiplier for the number of channels (default=1)
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['input_block'].update(dict(layout='cna', filters=32, kernel_size=3, strides=2))
        config['body'].update(_DEFAULT_BODY_ARCH)
        config['head'].update(dict(layout='Vf'))
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['units'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

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
        sep_strides, double_filters, width_factor = \
            cls.pop(['strides', 'double_filters', 'width_factor'], kwargs)

        with tf.variable_scope(name):
            x = inputs
            for i, strides in enumerate(sep_strides):
                x = cls.block(x, strides=strides, double_filters=double_filters[i], width_factor=width_factor,
                              name='block-%d' % i, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, strides=1, double_filters=False, width_factor=1, name=None, **kwargs):
        """ A network building block consisting of a separable depthwise convolution and 1x1 pointwise covolution.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        strides : int
            strides in separable convolution
        double_filters : bool
            if True number of filters in 1x1 covolution will be doubled
        width_factor : float
            multiplier for the number of filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        data_format = kwargs.get('data_format')
        num_filters = cls.channels_shape(inputs, data_format) * width_factor
        filters = [num_filters, num_filters*2] if double_filters else num_filters
        return conv_block(inputs, 'sna cna', filters, [3, 1], name=name, strides=[strides, 1], **kwargs)
