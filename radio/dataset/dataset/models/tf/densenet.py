"""
Huang G. et al. "`Densely Connected Convolutional Networks
<https://arxiv.org/abs/1608.06993>`_"
"""
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class DenseNet(TFModel):
    """ DenseNet

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks'. See :meth:`.TFModel._make_inputs`.

    input_block : dict

    body : dict
        strides : list of int
            strides in separable convolutions

        double_filters : list of bool
            if True, number of filters in 1x1 covolution will be doubled

        width_factor : float
            multiplier for the number of channels (default=1)

        block : dict
            parameters for dense block, including :func:`~.layers.conv_block` parameters, as well as

            growth_rate : int
                number of output filters in each layer (default=32)

            bottleneck : bool
                whether to use 1x1 convolutions in each layer (default=True)

    transition_layer : dict
        parameters for transition layers, including :func:`~.layers.conv_block` parameters, as well as

        reduction_factor : float
            a multiplier for number of output filters (default=1)

    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['input_block'].update(dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                          pool_size=3, pool_strides=2))
        config['body']['block'] = dict(layout='nacd', dropout_rate=.2, growth_rate=32, bottleneck=True)
        config['body']['transition_layer'] = dict(layout='nacv', kernel_size=1, strides=1,
                                                  pool_size=2, pool_strides=2,
                                                  reduction_factor=1)
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
        num_blocks, block, transition = cls.pop(['num_blocks', 'block', 'transition_layer'], kwargs)
        block = {**kwargs, **block}
        transition = {**kwargs, **transition}

        with tf.variable_scope(name):
            x = inputs
            for i, num_layers in enumerate(num_blocks):
                x = cls.block(x, num_layers=num_layers, name='block-%d' % i, **block)
                if i < len(num_blocks) - 1:
                    x = cls.transition_layer(x, name='transition-%d' % i, **transition)
        return x

    @classmethod
    def block(cls, inputs, num_layers=3, name=None, **kwargs):
        """ A network building block consisting of a stack of 1x1 and 3x3 convolutions.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_layers : int
            number of conv layers
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        layout, growth_rate, bottleneck = \
            cls.pop(['layout', 'growth_rate', 'bottleneck'], kwargs)

        with tf.variable_scope(name):
            axis = cls.channels_axis(kwargs['data_format'])
            x = inputs
            for i in range(num_layers):
                block_concat = x
                if bottleneck:
                    x = conv_block(x, filters=growth_rate * 4, kernel_size=1, layout=layout,
                                   name='bottleneck-%d' % i, **kwargs)
                x = conv_block(x, filters=growth_rate, kernel_size=3, layout=layout,
                               name='conv-%d' % i, **kwargs)
                x = tf.concat([block_concat, x], axis=axis)
            x = tf.identity(x, name='output')
        return x

    @classmethod
    def transition_layer(cls, inputs, name='transition_layer', **kwargs):
        """ An intermediary interconnect layer between two dense blocks

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
        kwargs = cls.fill_params('body/transition_layer', **kwargs)
        reduction_factor = cls.get('reduction_factor', kwargs)
        num_filters = cls.channels_shape(inputs, kwargs.get('data_format'))
        return conv_block(inputs, filters=num_filters * reduction_factor, name=name, **kwargs)


class DenseNet121(DenseNet):
    """ The original DenseNet-121 architecture

    References
    ----------
    .. Huang G. et al. "r "Densely Connected Convolutional Networks""
       Arxiv.org `<https://arxiv.org/abs/1608.06993>`_
    """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_blocks'] = [6, 12, 24, 32]
        return config

class DenseNet169(DenseNet):
    """ The original DenseNet-169 architecture

    References
    ----------
    .. Huang G. et al. "r "Densely Connected Convolutional Networks""
       Arxiv.org `<https://arxiv.org/abs/1608.06993>`_
    """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_blocks'] = [6, 12, 32, 16]
        return config

class DenseNet201(DenseNet):
    """ The original DenseNet-201 architecture

    References
    ----------
    .. Huang G. et al. "r "Densely Connected Convolutional Networks""
       Arxiv.org `<https://arxiv.org/abs/1608.06993>`_
    """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_blocks'] = [6, 12, 48, 32]
        return config

class DenseNet264(DenseNet):
    """ The original DenseNet-264 architecture

    References
    ----------
    .. Huang G. et al. "r "Densely Connected Convolutional Networks""
       Arxiv.org `<https://arxiv.org/abs/1608.06993>`_
    """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_blocks'] = [6, 12, 64, 48]
        return config
