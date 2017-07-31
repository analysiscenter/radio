"""Helper functions for creating layers """

import tensorflow as tf

def selu(x):
    """
    selu activation function
    Args:
        input tensor x
    Return:
        selu(x)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def conv3d_bnorm_activation(inputs, training, add_bnorm=True,
                            activation=tf.nn.relu, kernel=(7, 7, 7), channels=1,
                            initializer='xavier'):
    """
    form conv3d -> batch norm -> relu
    Args:
        inputs: input layer, 5d-tensor: (None, depth, width, height, input_channels)
        training: boolean tf placeholder; if True then bnorm is used in training mode
        add_bnorm: whether Batch-norm should be added after convolution
        activation: nonlinearity to use after bnorm
        kernel: conv3d filter size
        channels: number of channels in image after conv
        initializer: initializer to use for weights initialization
            can be 'xavier' or 'normal'
    Return:
        output, 5d-tensor: (None, depth, width, height, channels)
    """
    # set initializer
    if initializer == 'xavier':
        init = tf.contrib.layers.xavier_initializer()
    elif initializer == 'normal':
        init = tf.random_normal_initializer()
    conv = tf.layers.conv3d(inputs, filters=channels, kernel_size=kernel, padding='same',
                            name='convolution', kernel_initializer=init)

    # batch norm if needed
    if add_bnorm:
        normed = tf.layers.batch_normalization(
            conv, training=training, name='batch-norm')
    else:
        normed = conv

    # activation if needed
    if activation is None:
        return normed
    else:
        return activation(normed)


def deconv3d_bnorm_activation(inputs, training, add_bnorm=True,
                              activation=tf.nn.relu, kernel=(5, 5, 5), channels=1,
                              initializer='xavier'):
    """
    form upsampling deconv3d -> batch norm -> relu

    *Note: this is doubling deconv-layer
        that is, the stride is always 2 in all spatial dims

    Args:
        inputs: input layer, 5d-tensor: (None, depth, width, height, input_channels)
        training: boolean tf placeholder; if True then bnorm is used in training mode
        add_bnorm: whether Batch-norm should be added after deconvolution
        activation: nonlinearity to use after bnorm
        kernel: deconv3d filter size
        channels: number of channels in image after deconv
        initializer: initializer to use for weights initialization
            can be 'xavier' or 'normal'

    Return:
        output, 5d-tensor: (None, 2 * depth, 2 * width, 2 * height, channels)
    """
    # set output shape (double input shape in 3 spatial dims)
    input_shape = inputs.shape
    inp_shape = tf.shape(inputs)
    output_shape = ([inp_shape[0]] +
                    [2 * int(input_shape[i]) for i in range(1, len(input_shape) - 1)] +
                    [channels])
    # set strides for doubling
    strides = [1, 2, 2, 2, 1]

    # set initializer

    if initializer == 'xavier':
        init = tf.contrib.layers.xavier_initializer()
    elif initializer == 'normal':
        init = tf.random_normal_initializer()

    # create filter variable
    ftr = tf.get_variable('filter', shape=list(kernel) + [channels] +
                          [int(input_shape[-1])], initializer=init)

    # apply deconvolution
    deconv = tf.nn.conv3d_transpose(inputs, ftr, output_shape, strides,
                                    padding='SAME', name='deconvolution')
    # add bias, init by zeroes
    bias = tf.get_variable('bias', shape=[channels],
                           initializer=tf.zeros_initializer())
    deconv = tf.nn.bias_add(deconv, bias)

    # bnorm if needed
    if add_bnorm:
        normed = tf.layers.batch_normalization(
            deconv, training=training, name='batch-norm')
    else:
        normed = deconv

    # activation if needed
    if activation is None:
        return normed
    else:
        return activation(normed)


def vnet_up(scope, net_up, net_down, training, add_bnorm=True, activation=tf.nn.relu,  # pylint: disable=too-many-arguments
            kernel=(3, 3, 3), channels=1, initializer='xavier'):
    """
    form upsampling vnet layer

    Args:
        scope: scope to create ops in
        net_up: layer from the right (upsampling) part of v-net,
            5d-tensor of shape (None, depth, width, height, input_channels)
        net_down: layer from the left (downsampling) part of v-net
            should shape = net_up.shape
        training: boolean tf placeholder; if True then bnorm is used in training mode
        add_bnorm: whether Batch-norm should be added after deconvolution
        activation: nonlinearity to use after bnorm
        kernel: filter size of deconvolution
        channels: number of channels in output layer, typically = 1/2 x channels in
            net_down
        initializer: initializer to use for weights initialization
            can be 'xavier' or 'normal'

    Return:
        5d-tensor, shape[4] = channels; with name = 'scope/output'

        Schematically:

                                                output
                    ----                         ----
                      net_down               net_up   deconv(net_down & net_up) /|\
                     --------    concat     --------
                      ------------ ... ------------
                                   ...
    """
    params = dict(training=training, add_bnorm=add_bnorm, activation=activation,
                  kernel=kernel, channels=channels, initializer=initializer)

    with tf.variable_scope(scope):   # pylint: disable=not-context-manager
        concatted = tf.concat([net_up, net_down], 4)
        output = deconv3d_bnorm_activation(inputs=concatted, **params)
        output = tf.identity(output, name='output')

    return output

def vnet_down(scope, net_down, training, pool_size=(2, 2, 2), strides=(2, 2, 2), **kwargs):
    """
    form downsampling vnet layer

    Args:
        scope: scope to create ops in
        net_down: layer from the left (downsampling) part of v-net
        training: boolean tf placeholder; if True then bnorm is used in training mode
        pool_size: window for downsampling max-pooling
        strides: max_pooling strides
        add_bnorm: whether Batch-norm should be added after deconvolution
        activation: nonlinearity to use after bnorm
        kernel: filter size of convolution
        channels: number of channels in output layer, typically = 2 x channels in
            net_down
        initializer: initializer to use for weights initialization
            can be 'xavier' or 'normal'

    Return:
        output layer: 5d-tensor; shape[4] = channels; with name = 'scope/output'


    Schematically:
                    net_down -> conv -> max pooling
                    ----                         ----
                     output
                     --------               --------
                      ------------ ... ------------
                                   ...
    """
    with tf.variable_scope(scope):                     # pylint: disable=not-context-manager
        conv = conv3d_bnorm_activation(inputs=net_down, training=training, **kwargs)
        output = tf.layers.max_pooling3d(conv, pool_size, strides)
        output = tf.identity(output, name='output')

    return output


def tf_dice_loss(scope, masks_prediction, masks_ground_truth, epsilon=0):
    """ Form loss = - dice given predicions for masks and true masks. Resulting
            loss is given by tf-tensor.

    Args:
        scope: scope to create loss-op
        masks_prediction: normalized to [0, 1] 5d-tensor output
            of a net (one dim is fake)
        masks_ground_truth: true cancer-masks, 5d-placeholder with
            last fake dim
        epsilon: add small epsilon to the denominator if problems with
            division on zero arise
    Return:
        zero-shape tf tensor = mean(dice loss) across batch of scans;
            name = 'scope/loss'
    """
    # compute dice = 2 |A*B| / (|A| + |B| + epsilon)
    with tf.variable_scope(scope):                    # pylint: disable=not-context-manager
        sum_preds = tf.reduce_sum(masks_prediction)
        sum_truth = tf.reduce_sum(masks_ground_truth)
        sum_intersection = tf.reduce_sum(masks_ground_truth * masks_prediction)
        dice = 2 * (sum_intersection + epsilon) / (sum_truth + sum_preds + 2 * epsilon)

        # loss = -dice
        loss = tf.multiply(dice, -1.0, name='loss')

    return loss

