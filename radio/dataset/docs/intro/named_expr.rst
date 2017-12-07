
Named expressions
=================
As pipelines are declarative you might need a way to address data which exists only when the pipeline is executed.
For instance, model training takes batch data, but you don't have any batches when you declare a pipeline.
Batches appear only when the pipeline is run. This is where **named expressions** come into play.

A named expression specifies a substitution rule. When the pipeline is being executed,
a named expression is replaced with the value calculated according to that rule.

There are 4 types of named expressions:

* B('name') - a batch class attribute or component name
* V('name') - a pipeline variable name
* C('name') - a pipeline config option
* F(name) - a callable which takes a batch (could be a batch class method or an arbitrary function)


B - batch component
-------------------
::

    pipeline
        ...
        .train_model(model_name, feed_dict={'features': B('features'), 'labels': B('labels')})
        ...

At each iteration ``B('features')`` and ``B('labels')`` will be replaced with ``current_batch.features``
and ``current_batch.labels``, i.e. `batch components <components>`_ or attributes.


V - pipeline variable
---------------------
::

    pipeline
        ...
        .train_model(V('model_name'), ...)
        ...

At each iteration ``V('model_name')`` will be replaced with the current value of ``pipeline.get_variable('model_name')``.


C - config option
-----------------
::

    config = dict(model=ResNet18, model_config=model_config)

    train_pipeline = dataset.train.pipeline(config)
        ...
        .init_model('dynamic', C('model'), 'my_model', C('model_config'))
        ...

At each iteration ``C('model')`` will be replaced with the current value of ``pipeline.config['model']``.

This is an example of a model independent pipeline which allows to change models, for instance,
to assess performance of various models.


F - callable
------------
A function which takes a batch and, possibly, other arguments.

It can be a lambda function::

    pipeline
        .init_model('dynamic', MyModel, 'my_model', config={
            'inputs': {'images': {'shape': F(lambda batch: batch.images.shape[1:])}}
        })

or a batch class method::

    pipeline
        .train_model(model_name, make_data=F(MyBatch.pack_to_feed_dict, task='segmentation'))

or a function::

    def get_boxes(batch, shape):
        x_coords = slice(0, shape[0])
        y_coords = slice(0, shape[1])
        return batch.images[:, y_coords, x_coords]

    pipeline
        ...
        .update_variable(var_name, F(get_boxes, V('image_shape')))
        ...

or any other Python callable.


.. note:: Most of the time the first parameter passed to ``F``-function contains the current batch.
   However, there are a few exceptions.

As static models are initialized before a pipeline is run (i.e. before any batch is created),
all ``F``-functions specified in static ``init_model`` get ``pipeline`` as a first parameter.

In ``train_model`` and ``predict_model`` ``F``-functions take the batch as the first parameter and the model
as the second parameter.
