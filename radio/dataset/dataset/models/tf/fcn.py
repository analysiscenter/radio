"""
Shelhamer E. et al "`Fully Convolutional Networks for Semantic Segmentation
<https://arxiv.org/abs/1605.06211>`_"
"""
import tensorflow as tf

from . import TFModel, VGG16
from .layers import conv_block


class FCN(TFModel):
    """ Base Fully convolutional network (FCN) """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['common']['dropout_rate'] = .5
        config['input_block']['base_network'] = VGG16
        config['body']['filters'] = 100
        config['body']['upsampling_kernel'] = 3

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        config['body']['num_classes'] = self.num_classes('targets')
        config['head']['num_classes'] = self.num_classes('targets')
        config['head']['original_images'] = config['input_block']['inputs']

        return config

    @classmethod
    def input_block(cls, inputs, base_network, name='input_block', **kwargs):
        """ Base network

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        base_network : class
            base network class
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = base_network.input_block(inputs, name='input_block', **kwargs)
            x = base_network.body(x, name='body', **kwargs)
        return x

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor

        Returns
        -------
        tf.Tensor
        """
        raise NotImplementedError()

    @classmethod
    def head(cls, inputs, original_images, num_classes, name='head', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        original_images : tf.Tensor
            the tensor with source images (provide the shape to upsample to)
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        factor = cls.pop('factor', kwargs)

        x = conv_block(inputs, filters=num_classes, kernel_size=factor, layout='t', name=name,
                       **{**kwargs, 'strides': factor})
        x = cls.crop(x, original_images, kwargs.get('data_format'))
        return x


class FCN32(FCN):
    """  Fully convolutional network (FCN32)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    input_block : dict
        base_network : class
            base network (VGG16 by default)

    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
        upsampling_kernel : int
            kernel_size for upsampling (default=3)

    head : dict
        factor : int
            upsampling factor (default=32)
    """
    @classmethod
    def default_config(cls):
        config = FCN.default_config()
        config['head']['factor'] = 32
        return config

    @classmethod
    def body(cls, inputs, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        _ = num_classes
        kwargs = cls.fill_params('body', **kwargs)
        filters = kwargs.pop('filters')
        layout = kwargs.pop('layout', 'cnad cnad')
        return conv_block(inputs, filters=filters, kernel_size=[7, 1], layout=layout, name=name, **kwargs)


class FCN16(FCN):
    """  Fully convolutional network (FCN16)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    input_block : dict
        base_network : class
            base network (VGG16 by default)
        skip_name : str
            tensor name for the skip connection.
            Default='block-3/output:0' for VGG16.

    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
        upsampling_kernel : int
            kernel_size for upsampling (default=3)

    head : dict
        factor : int
            upsampling factor (default=16)
    """
    @classmethod
    def default_config(cls):
        config = FCN.default_config()
        config['head']['factor'] = 16
        config['input_block']['skip_name'] = '/input_block/body/block-3/output:0'
        return config

    @classmethod
    def input_block(cls, inputs, name='input_block', **kwargs):
        kwargs = cls.fill_params('input_block', **kwargs)

        x = FCN.input_block(inputs, name=name, **kwargs)
        skip_name = tf.get_default_graph().get_name_scope() + kwargs['skip_name']
        skip = tf.get_default_graph().get_tensor_by_name(skip_name)
        return x, skip

    @classmethod
    def body(cls, inputs, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            two input tensors
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters, kernel = cls.pop(['filters', 'upsampling_kernel'], kwargs)

        with tf.variable_scope(name):
            x, skip = inputs
            x = FCN32.body(x, filters=filters, num_classes=num_classes, name='fcn32', **kwargs)
            x = conv_block(x, 't', filters=num_classes, kernel_size=kernel, name='fcn32_2', strides=2, **kwargs)

            skip = conv_block(skip, 'c', filters=num_classes, kernel_size=1, name='pool', **kwargs)
            x = cls.crop(x, skip, kwargs.get('data_format'))
            output = tf.add(x, skip, name='output')
        return output


class FCN8(FCN):
    """  Fully convolutional network (FCN8)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    input_block : dict
        base_network : class
            base network (VGG16 by default)
        skip1_name : str
            tensor name for the first skip connection.
            Default='block-3/output:0' for VGG16.
        skip2_name : str
            tensor name for the second skip connection.
            Default='block-2/output:0' for VGG16.

    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
        upsampling_kernel : int
            kernel_size for upsampling (default=3)

    head : dict
        factor : int
            upsampling factor (default=8)
    """
    @classmethod
    def default_config(cls):
        config = FCN.default_config()
        config['head']['factor'] = 8
        config['input_block']['skip1_name'] = '/input_block/body/block-3/output:0'
        config['input_block']['skip2_name'] = '/input_block/body/block-2/output:0'
        return config

    @classmethod
    def input_block(cls, inputs, name='input_block', **kwargs):
        kwargs = cls.fill_params('input_block', **kwargs)
        x = FCN.input_block(inputs, name=name, **kwargs)
        skip1_name = tf.get_default_graph().get_name_scope() + kwargs['skip1_name']
        skip1 = tf.get_default_graph().get_tensor_by_name(skip1_name)
        skip2_name = tf.get_default_graph().get_name_scope() + kwargs['skip2_name']
        skip2 = tf.get_default_graph().get_tensor_by_name(skip2_name)
        return x, skip1, skip2

    @classmethod
    def body(cls, inputs, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters, kernel = cls.pop(['filters', 'upsampling_kernel'], kwargs)

        with tf.variable_scope(name):
            x, skip1, skip2 = inputs

            x = FCN16.body((x, skip1), filters=filters, num_classes=num_classes, name='fcn16', **kwargs)
            x = conv_block(x, 't', num_classes, kernel, name='fcn16_2', strides=2, **kwargs)

            skip2 = conv_block(skip2, 'c', num_classes, 1, name='pool2')

            x = cls.crop(x, skip2, kwargs.get('data_format'))
            output = tf.add(x, skip2, name='output')
        return output
