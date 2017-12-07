===================================
Welcome to Dataset's documentation!
===================================
`Dataset` helps you conveniently work with random or sequential batches of your data
and define data processing and machine learning workflows even for datasets that do not fit into memory.

Main features:

* flexible batch generaton
* deterministic and stochastic pipelines
* datasets and pipelines joins and merges
* data processing actions
* flexible model configuration
* within batch parallelism
* batch prefetching
* ready to use ML models and proven NN architectures
* convenient layers and helper functions to build custom models.


Contents
========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   intro/intro
   intro/classes
   api/dataset


Basic usage
===========
::

    my_workflow = my_dataset.pipeline()
                    .load('/some/path')
                    .do_something()
                    .do_something_else()
                    .some_additional_action()
                    .save('/to/other/path')

The trick here is that all the processing actions are lazy. They are not executed until their results are needed, e.g. when you request a preprocessed batch::

    my_workflow.run(BATCH_SIZE, shuffle=True, n_epochs=5)

or ::

    for batch in my_workflow.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=5):
        # only now the actions are fired and data is being changed with the workflow defined earlier
        # actions are executed one by one and here you get a fully processed batch

or ::

    NUM_ITERS = 1000
    for i in range(NUM_ITERS):
        processed_batch = my_workflow.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
        # only now the actions are fired and data is changed with the workflow defined earlier


Train a neural network
======================
Dataset includes ready-to-use proven architectures like VGG, Inception, ResNet and many others.
To apply them to your data just choose a model, specify the inputs (like the number of classes or images shape)
and call ``train_model``. Of course, you can also choose a loss function, an optimizer and many other parameters, if you want.::

    from dataset.models.tf import ResNet34

    my_workflow = my_dataset.pipeline()
                  .init_model('dynamic', ResNet34, config={
                              'inputs': {'images': {'shape': B('image_shape')},
                                         'labels': {'classes': 10, 'transform': 'ohe', 'name': 'targets'}},
                              'input_block/inputs': 'images'})
                  .load('/some/path')
                  .some_transform()
                  .another_transform()
                  .train_model('ResNet34', feed_dict={'images': B('images'), 'labels': B('labels')})
                  .run(BATCH_SIZE, shuffle=True)


Installation
============


Python package
--------------

With modern `pipenv <https://docs.pipenv.org/>`_::

    pipenv install git+https://github.com/analysiscenter/dataset.git#egg=dataset

With old-fashioned `pip <https://pip.pypa.io/en/stable/>`_::

    pip3 install git+https://github.com/analysiscenter/dataset.git


After that just import `dataset`::

    import dataset as ds


Git submodule
-------------

.. note:: `Dataset` module is in the beta stage. Your suggestions and improvements are very welcome.

.. note:: `Dataset` supports python 3.5 or higher.


In many cases it is much more convenient to install `dataset` as a submodule in your project repository than as a python package::

    git submodule add https://github.com/analysiscenter/dataset.git
    git submodule init
    git submodule update


If your python file is located in another directory, you might need to add a path to `dataset` submodule location::

    import sys
    sys.path.insert(0, "/path/to/dataset")
    import dataset as ds

What is great about using a submodule is that every commit in your project can be linked to its own commit of a submodule.
This is extremely convenient in a fast paced research environment.

Relative import is also possible::

    from .dataset import Dataset



Citing Dataset
==============
Please cite Dataset in your publications if it helps your research.

.. image:: https://zenodo.org/badge/84835419.svg
    :target: https://zenodo.org/badge/latestdoi/84835419

::

    Roman Kh et al. Dataset library for fast ML workflows. 2017. doi:10.5281/zenodo.1041203

::

    @misc{roman_kh_2017_1041203,
      author       = {Roman Kh and et al},
      title        = {Dataset library for fast ML workflows},
      year         = 2017,
      doi          = {10.5281/zenodo.1041203},
      url          = {https://doi.org/10.5281/zenodo.1041203}
    }

