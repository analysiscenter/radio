# pylint: disable=too-many-arguments
# pylint: disable=not-context-manager
""" Implementation of custom volumetric network for lung cancer detection. """

import numpy as np
import tensorflow as tf
from ..layers import bn_dilated_conv3d
from ..utils import repeat_tensor
from ....dataset.dataset.models.tf.layers import conv_block
from ....dataset.dataset.models.tf import TFModel

class DilatedNoduleNet(TFModel):
    """ Implementation of custom encoder-decoder architecture with dilated convolutions.

    Architecture is inspired by VNet (Milletari et al., https://arxiv.org/abs/1606.04797).

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    body : dict
        num_blocks : int
            number of encoder/decoder blocks (default=4)

        filters : list of int
            number of filters in each block (default=[128, 256, 512, 1024])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        """ Default config. """
        config = TFModel.default_config()

        filters = 32

        config['input_block'].update({})
        config['body']['upsampling_kernel'] = 3
        config['body']['num_blocks'] = 4
        config['body'][
            'filters'] = 2 ** np.arange(config['body']['num_blocks']) * filters * 2
        config['body']['dilation_rate'] = [1, 2]
        config['body']['dilation_share'] = [0.5, 0.5]
        config['body']['upsampling_mode'] = 'deconv'
        return config

    def build_config(self, names=None):
        """ Build config. """
        config = super().build_config(names)
        config['head']['num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def dilated_branches(cls, inputs, filters, kernel_size, dilation_rate, name,
                         activation=tf.nn.relu, padding='same', is_training=True):
        """ Convolutional block with parallel branches having different dilation rate.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple(int, ...)
            number of filters corresponding to branches with different dilation rate.
        kernel_size : tuple(int, ...)
            size of convolutional kernel corresponding to branches
            with different dilation rate. Kernel size is considered
            to be the same along [zyx] dimensions.
        dilation_rate : tuple(int, ...)
            dilation rate for convolutional branches. Dilation rate is considered
            to be the same along [zyx] dimensions.
        activation : tensorflow activation function
        padding : str
            padding to use in convolution operation. Can be 'same' or 'valid'.
        is_training : bool or bool tensor
            indicator of training or prediction mode.
        name : str
            name of the block that will be used as argument of tf.variable_scope.

        Returns
        -------
        tf.Tensor
        """

        if not all(isinstance(arg, (tuple, list)) for arg in (filters, kernel_size, dilation_rate)):
            raise ValueError("Arguments 'filters', 'kernel_size', 'dilation_rate' "
                             +"must be tuples or lists")

        branches = []
        with tf.variable_scope(name):
            for f, k, d in zip(filters, kernel_size, dilation_rate):

                if not isinstance(k, (tuple, list)):
                    _ksize = [k] * 3
                else:
                    _ksize = k

                if not isinstance(d, (tuple, list)):
                    _drate = [d] * 3
                else:
                    _drate = d

                b = bn_dilated_conv3d(inputs, f, _ksize, padding=padding,
                                      is_training=is_training, activation=activation,
                                      dilation=_drate, name='conv3d_rate_{}'.format(_drate[0]))
                branches.append(b)
            outputs = tf.concat(branches, axis=4)

        return outputs

    @classmethod
    def decoder_block(cls, inputs, filters, name, **kwargs):
        """ 3x3 convolution and 2x2 transposed convolution or upsampling

        Each of two 3x3x3 convolutions contains several branches with
        different dilation rate.

        Parameters
        ----------
        inputs : tuple(tf.Tensor, tf.Tensor)
            two input tensors
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
        is_training = cls.pop('is_training', config)

        mode = cls.pop('upsampling_mode', config)  # Added
        dilation_rate = cls.pop('dilation_rate', config)  # Added
        dilation_share = np.asarray(cls.pop('dilation_share', config))
        dilation_share /= dilation_share.sum()
        _filters = np.rint(filters * dilation_share).astype(np.int).tolist()

        if kwargs.get('data_format') == 'channels_last':
            repeat_times = (1, 2, 2, 2, 1)
            axis = -1
        else:
            repeat_times = (1, 1, 2, 2, 2)
            axis = 1

        with tf.variable_scope(name):
            x, skip = inputs

            if mode == 'deconv':
                conv_kwargs = dict(filters=filters, kernel_size=kernel,
                                   strides=2, activation=tf.nn.relu,
                                   use_bias=False, is_training=is_training)
                x = conv_block(x, 'tna', {**kwargs, **conv_kwargs})
            elif mode == 'repeat':
                x = repeat_tensor(x, repeat_times)

            x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
            x = tf.concat((skip, x), axis=axis)

            x = cls.dilated_branches(x, _filters, (3, 3, 3), dilation_rate,
                                     name='conv_I', is_training=is_training)

            x = cls.dilated_branches(x, _filters, (3, 3, 3), dilation_rate,
                                     name='conv_II', is_training=is_training)
        return x

    @classmethod
    def encoder_block(cls, inputs, filters, name, **kwargs):
        """ Two 3x3x3 convolutions and 2x2x2 max pooling with stride 2.

        Each of two 3x3x3 convolutions contains several branches with
        different dilation rate.

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
        is_training = cls.pop('is_training', config)

        dilation_rate = cls.pop('dilation_rate', config)

        dilation_share = np.asarray(cls.pop('dilation_share', config))
        dilation_share /= dilation_share.sum()
        _filters = np.rint(filters * dilation_share).astype(np.int).tolist()
        with tf.variable_scope(name):
            x = cls.dilated_branches(inputs, _filters, (3, 3, 3), dilation_rate,
                                     name='conv_I', is_training=is_training)

            x = cls.dilated_branches(x, _filters, (3, 3, 3), dilation_rate,
                                     name='conv_II', is_training=is_training)

            downsampled_x = tf.layers.max_pooling3d(x, pool_size=(2, 2, 2),
                                                    strides=(2, 2, 2),
                                                    padding='same',
                                                    name='max_pool3d')
        return x, downsampled_x

    @classmethod
    def central_block(cls, inputs, filters, name, **kwargs):
        """ Block that situated between encoder and decoder branches.

        Block consists of 1x1x1 followed by 3x3x3. Note that 3x3x3 convolution
        can contain several branches with different dilation rate.

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
        is_training = cls.pop('is_training', config)

        dilation_rate = cls.pop('dilation_rate', config)

        dilation_share = np.asarray(cls.pop('dilation_share', config))
        dilation_share /= dilation_share.sum()
        _filters = np.rint(filters * dilation_share).astype(np.int).tolist()

        with tf.variable_scope(name):
            x =  conv_block(inputs, 'cna', filters=filters, kernel_size=1,
                            activation=tf.nn.relu, is_training=is_training)

            x = cls.dilated_branches(x, _filters, (3, 3, 3), dilation_rate,
                                     name='conv3D_dilated', is_training=is_training)
        return x

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in encoder_block
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
            for i, ifilters in enumerate(filters[:-1]):
                y, x = cls.encoder_block(
                    x, ifilters, name='encoder-'+str(i), **kwargs)
                encoder_outputs.append(y)

            x = cls.central_block(
                x, filters[-1], name='central_block', **kwargs)

            for i, ifilters in enumerate(filters[:-1][::-1]):
                x = cls.decoder_block((x, encoder_outputs[-i-1]), ifilters//2,
                                      name='decoder-'+str(i), **kwargs)
        return x

    @classmethod
    def head(cls, inputs, num_classes, name='head', **kwargs):
        """ Conv block followed by 1x1 convolution.

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
        pred_kwargs = dict(filters=num_classes, kernel_size=1,
                           activation=tf.nn.sigmoid, layout='cna')
        with tf.variable_scope(name):
            x = conv_block(inputs, name='conv', **kwargs)
            x = conv_block(x, **pred_kwargs)
        return x
