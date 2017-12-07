Tensorflow layers and losses
============================

Convolution block
-----------------
The module :mod:`.models.tf.layers` includes a :func:`convolution building block <.models.tf.layers.conv_block>`
which helps building complex networks in a concise way.

The advantages of using ``conv_block`` are:

- it helps to create more sophisticated networks with fewer lines of code;
- it allows to build multidimensional models with the same code (namely 1d, 2d, and 3d);
- it contains new layers missing in TensorFlow (e.g. separable 1d and 3d convolutions, 1d transposed convolutions, mip);
- it uses a fast CuDNN implementation of batch norm;

The block consist of predefined layers, among which:

- convolutions (as well as dilated, separable and transposed convolutions)
- batch normalization
- activation
- global and spatial max pooling
- global and spatial average pooling
- maximum intensity projection (mip)
- dropout

The layers types and order are set by ``layout`` parameter. Thus, for instance, you can easily create
a sequence of 4 layers (3x3 convolution, batch norm, relu and max pooling) in one line of code::

    x = conv_block(x, layout='cnap', filters=32, kernel_size=3, name='conv1', training=self.is_training)


Or a more sophisticated example - a full 14-layer VGG-like model in just 6 lines::

    class MyModel(TFModel):
        def body(self, inputs, **kwargs):
            num_classes = self.num_classes('labels')

            x = inputs
            x = conv_block(x, 'cacap', 64, 3, name='block1', **kwargs)
            x = conv_block(x, 'cacap', 128, 3, name='block2', **kwargs)
            x = conv_block(x, 'cacacap', 256, 3, name='block3', **kwargs)
            x = conv_block(x, 'cacacap', 512, 3, name='block4', **kwargs)
            x = conv_block(x, 'cacacap', 512, 3, name='block5', **kwargs)
            x = conv_block(x, num_classes, 3, layout='cP', name='classification', **kwargs)
            output = tf.identity(x, name='predictions')
            return output

That's a fully working example. Just try it with a simple pipeline:

.. code-block:: python

    from dataset.openses import MNIST
    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block, global_average_pooling

    mnist = MNIST()

    train_pp = (mnist.train.p
                .init_variable('current_lost', 0)
                .init_model('dynamic', MyModel, 'conv',
                            config={'loss': 'ce',
                                    'inputs': dict(images={'shape': (28, 28, 1)},
                                                   labels={'classes': 10, 'dtype': 'uint8',
                                                           'transform': 'ohe', 'name': 'targets'}),
                                    'input_block/inputs': 'images'})
                .train_model('conv', fetches='loss', feed_dict={'images': B('images'),
                                                                'labels': B('labels')},
                             save_to=V('current_loss'), mode='a')
                .print_variable('current_loss')
                .run(128, shuffle=True, n_epochs=2))

When ``layout`` includes several layers of the same type, each one can have its own parameters,
if corresponding arguments are passed as lists.

A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between) is defined as follows::

    x = conv_block(x, 'cacac', [64, 64, 256], [1, 3, 1])

An even more complex block:

- 5x5 conv with 32 filters
- relu
- 3x3 conv with 32 filters
- relu
- 3x3 conv with 64 filters and a spatial stride 2
- relu
- batch norm
- dropout with rate 0.15

::

    x = conv_block(x, 'cacacand', [32, 32, 64], [5, 3, 3], strides=[1, 1, 2], dropout_rate=.15, training=self.is_training)

Or the earlier defined 14-layers VGG network as a one-liner::

    x = conv_block(x, 'cacap'*2 + 'cacacap'*3 + 'caP', [64]*2 + [128]*2 + [256]*3 + [512]*6 + [num_classes], 3)

However, in terms of training performance and prediction accuracy the following block with strided separable (grouped) convolutions and dropout will usually perform much better::

    x = conv_block(x, 'cna cna cna cnaP', [16, 32, 64, num_classes], 3, strides=[2, 2, 2, 1], dropout_rate=.15,
                   depth_multiplier=[1, 2, 2, 1], training=self.is_training)

Transposed convolution
-------------------------

.. autofunction:: dataset.models.tf.layers.conv_transpose
    :noindex:

.. autofunction:: dataset.models.tf.layers.conv1d_transpose
    :noindex:


Separable convolution
---------------------
.. autofunction:: dataset.models.tf.layers.separable_conv
    :noindex:

.. autofunction:: dataset.models.tf.layers.separable_conv1d
    :noindex:

.. autofunction:: dataset.models.tf.layers.separable_conv3d
    :noindex:


Pooling
-------

.. autofunction:: dataset.models.tf.layers.max_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.average_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.global_max_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.global_average_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.fractional_max_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.mip
    :noindex:

Flatten
-------
.. autofunction:: dataset.models.tf.layers.flatten
    :noindex:

.. autofunction:: dataset.models.tf.layers.flatten2d
    :noindex:


Maximum intensity projection
----------------------------
.. autofunction:: dataset.models.tf.layers.mip
    :noindex:
