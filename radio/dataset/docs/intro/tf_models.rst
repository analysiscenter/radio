Tensorflow models
=================

How to use a model
------------------
A model might be used for training or inference. In both cases you need to specify a model config and a pipeline.

A minimal config includes ``inputs`` and ``input_block`` sections::

    model_config = {
        'inputs': dict(images={'shape': (128, 128, 3)},
                       labels={'classes': 10, 'transform': 'ohe', 'name': 'targets'}),
        'input_block/inputs': 'images'
    }

Inputs section contains :meth:`a description of model input data <.TFModel._make_inputs>`, its shapes, transformations needed and names of the resulting tensors.

This config will create placeholders with the names ``images`` and ``labels``.
Later, these names will be used to feed data into the model when training or predicting.
Besides, one-hot encoding will be applied to ``labels`` and the encoded tensor will be named ``targets``.

Models based on :class:`.TFModel` expect that one of the inputs has a name ``targets`` (before or after transformations),
while ``input_block/inputs`` specifies which input (or inputs) will go through the network to turn into a tensor named ``predictions``.
Tensors with the names ``targets`` and ``predictions`` are used to define a model loss. Or you can write your own loss in a derived model.

Among other config options are ``loss``, ``optimizer``, ``decay``. Read :class:`.TFModel` documentations to find out more.

A minimal pipeline consists of ``init_model`` and ``train_model``::

    pipeline = my_dataset.p
        .init_model('dynamic', MyModel, 'my_model', model_config)
        .train_model('my_model', fetches='loss',
                     feed_dict={'images': B('images'),
                                'labels': B('labels')},
                     save_to=V('loss_history'), mode='a')
        .run(BATCH_SIZE, shuffle=True, n_epochs=5)


How to write a custom model
---------------------------

To begin with, take a look at `conv_block <tf_layers#convolution-block>`_ to find out how to write
complex networks in just one line of code. This block is a convenient building block for concise,
yet very expressive neural networks.

The simplest case you should avoid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Just redefine ``body()`` method.

For example, let's create a small fully convolutional network with 3 layers of 3x3 convolutions, batch normalization, dropout
and a dense layer at the end::

    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block

    class MyModel(TFModel):
        def body(self, inputs, **kwargs):
            x = conv_block(inputs, filters=[64, 128, 256], units=10, kernel_size=3,
                           layout='cna cna cna df', dropout_rate=.2, **kwargs)
            return x

Despite simplicity, this approach is highly discouraged as:

- the model parameters are hard coded in the body
- the model cannot be configured within a pipeline
- the model does not allow model composition, i.e. using this model components in other models.

The right way
~~~~~~~~~~~~~
Here we split network configuration and network definition into separate methods::

    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block

    class MyModel(TFModel):
        @classmethod
        def default_config(cls):
            config = TFModel.default_config()
            config['body'].update(dict(filters=[64, 128, 256], kernel_size=3, layout='cna cna cna'))
            config['head'].update(dict(units=2, layout='df', dropout_rate=.2))
            return config

        def build_config(self, names=None):
            config = super().build_config(names)
            config['head']['units'] = self.num_classes('targets')
            config['head']['filters'] = self.num_classes('targets')
            return config

        @classmethod
        def body(cls, inputs, name='body', **kwargs):
            kwargs = cls.fill_params('body', **kwargs)
            x = conv_block(inputs, **kwargs)
            return x

Note that ``default_config`` and ``body`` are ``@classmethods`` now, which means that they might be called without
instantiating a ``MyModel`` object.
This is needed for model composition, e.g. when ``MyModel`` serves as a base network for an FCN or SSD network.

On the other hand, ``build_config`` is an ordinary method, so it is called only when an instance of ``MyModel`` is created.

Thus, ``default_config`` should contain all the constants which are totaly independent of the dataset
and a specific task at hand, while ``build_config`` is intended to extract values from the dataset through pipeline's configuration
(for details see `Configuring a model <models#configuring-a-model>`_ and `TFModel configuration <#configuration>`_ below).

Now you can train the model with a simple pipeline::

    model_config = {
        'loss': 'ce',
        'decay': 'invtime',
        'optimizer': 'Adam',
        'inputs': dict(images={'shape': (128, 128, 3)},
                       labels={'shape': 10, 'transform': 'ohe', 'name': 'targets'}),
        'input_block/inputs': 'images'
    }

    pipeline = my_dataset.p
        .init_variable('loss_history', init_on_each_run=list)
        .init_model('dynamic', MyModel, 'my_model', model_config)
        .train_model('my_model', fetches='loss',
                     feed_dict={'images': B('images'),
                                'labels': B('labels')},
                     save_to=V('loss_history'), mode='a')
        .run(BATCH_SIZE, shuffle=True, n_epochs=5)

To switch to a fully convolutional head with 3x3 convolutions and global average pooling,
just add 1 line to the config::

    model_config = {
        ...
        'head/layout': 'cV'
    }

As a result, the very same model class might be used

- in numerous scenarios
- with different configurations
- for various tasks
- with heterogenous data.


Model structure
~~~~~~~~~~~~~~~
A model comprises of

- input_block
- body (which, in turn, might include blocks)
- head.

This division might seem somewhat arbitrary, though, many modern networks follow it.

input_block
'''''''''''
This block just transforms the raw inputs into more managable and initially preprocessed tensors.

Some networks do not need this (like VGG). However, most network have 1 or 2 convolutional layers
and sometimes also a max pooling layer with stride 2. These layers can be put into body, as well.
But the input block takes all irregular front layers, thus allowing for a regular body structure.


body
''''
Body contains a repetitive structure of building blocks. Most networks (like VGG, ResNet and the likes) have a straight sequence of blocks, while others (e.g. UNet or LinkNet) look like graphs with many interconnections.

Input block's output goes into body as inputs.
And body's output is a compressed representation (embedding) of the input tensors.
It can later be used for various tasks: classification, regression, detection, etc.
So ``body`` produces a task-independent embedding.


block
'''''
The network building block reflects the model's unique logic and specific technology.

Not surprisingly, many networks comprise different types of blocks, for example:

- UNet and LinkNet have downsampling and upsampling blocks
- Inception includes inception, reduction, and expanded blocks
- DenseNet have dense and transition blocks
- SqueezeNet alternates fire blocks with max-pooling.

You can have as many block types as you need.
Nevertheless, aim to make them universal and reusable elsewhere.
For instance, LinkNet uses ResNet's blocks.


head
''''
It receives body's output and produces a task-specific result, for instance, class logits for classification.
The default head consists of one :func:`.conv_block`. So, by specifying a model's config you can
instantiate models for different tasks.

Classification with 10 classes::

    config = {
        ...
        'loss': 'ce',
        'inputs': dict(images={'shape': (128, 128, 3)},
                       labels={'classes': 10, 'transform': 'ohe', 'name': 'targets'})
        'head': dict(layout='cdV', filters=10, dropout_rate=.2),
        'input_block/inputs': 'images'
    }

Regression::

    config = {
        ...
        'loss': 'mse',
        'inputs': dict(heart_signals={'shape': (4000, 1)},
                       targets={'shape': 1})
        'head': dict(layout='df', units=1, dropout_rate=.2),
        'input_block/inputs': 'heart_signals'
    }

Configuration
-------------

.. autoclass:: dataset.models.tf.TFModel
    :noindex:


How to configure model inputs
-----------------------------
.. automethod:: dataset.models.tf.TFModel._make_inputs
    :noindex:

Ready to use models
-------------------
.. toctree::

    tf_models_zoo
