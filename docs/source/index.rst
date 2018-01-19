.. RadIO documentation master file, created by
   sphinx-quickstart on Mon Nov  6 23:59:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RadIO's documentation!
=================================

**RadIO** is a framework for batch-processing of computed
tomography (CT)-scans for deep learning experiments.

Main features:

* Asynchronously load files from **DICOM** or **MetaImage** (mhd/raw)
* Dump files to `blosc <http://blosc.org/>`_ to reduce size of datasets and accelerate load
* Resize CT-scans in parallel using PIL or Scipy for faster preprocessing
* Set up workflows with minimum code with :func:`actions <dataset.action>` or use ready-made :doc:`pipelines <intro/pipelines>` for preprocessing or training models
* Train with ease a zoo of 18 high-performing neural networks for classification or segmentation
* Sample crops of any size from CT-scans for training, predict `on the whole scan <https://analysiscenter.github.io/lung_cancer//api/masked_batch.html#radio.preprocessing.ct_masked_batch.CTImagesMaskedBatch.predict_on_scan>`_
* Customize distribution of crop locations for improving training

This documentation is made to provide a comprehensive review of RadIO's capabilities. Nevertheless,
we advise you to take a look at the `tutorials-section <https://github.com/analysiscenter/radio/tree/master/tutorials>`_ on
our `repo <https://github.com/analysiscenter/radio>`_.

Tutorials
=========

In all, there are four tutorials:

* In the `first <https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.I.ipynb>`_ one you can learn how to set up a dataset of CT-scans and
set up a basic preprocessing workflow.
* In the `second tutorial <https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.II.ipynb>`_ we discuss in depth preprocessing and augmenting capabilities of
RadIO.
* The `third tutorial <https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.III.ipynb>`_ aims to explain how RadIO can help you to generate batches of crops,
that can be used for training a segmenting net on task of cancer segmentation.
* Lastly, in the `fourth tutorial <https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.IV.ipynb>`_
 you will train a set of segmenting neural networks (including `V-Net <https://arxiv.org/abs/1606.04797>`_)
from RadIO's zoo of models.

Contents
========
.. toctree::
   :maxdepth: 2

   intro/preprocessing
   intro/pipelines
   intro/models
   api/api


Preprocess scans using implemented actions
------------------------------------------

Preprocessing-module contains a set of :func:`actions <dataset.action>`,
that allow to efficiently prepare a dataset of CT-scans for training neural nets.
Say, you have a bunch of scans in **DICOM** format with varying shapes.
First, you index the scans using the :mod:`dataset <http://analysiscenter.github.io/dataset/>`.

.. code-block:: python

    from radio import CTImagesBatch
    from dataset import FilesIndex, Dataset

    dicomix = FilesIndex(path='path/to/dicom/*', no_ext=True) # set up the index
    dicomset = Dataset(index=dicomix, batch_class=CTImagesBatch) # init the dataset of dicom files

You may want to resize the scans to equal shape **[128, 256, 256]**,
normalize voxel densities to range **[0, 255]** and dump transformed
scans. Preprocessing like this can be easily done with the following
:class:`pipeline <dataset.Pipeline>` , just like that:

.. code-block:: python

    prep_ppl = (
        dicomset
        .pipeline()
        .load(fmt='dicom')
        .resize(shape=(128, 256, 256))
        .normalize_hu()
        .dump('/path/to/preprocessed/scans/')
    )
    prep_ppl.run(batch_size=20)

See the :doc:`documentation <intro/preprocessing>` for the description of
preprocessing actions implemented in the module.

Preprocess scans using a workflow from the box
----------------------------------------------

Pipelines-module contains ready-to-use workflows for most frequent tasks.
E.g. if you want to preprocess dataset of scans named ``ctset`` and
prepare data for training a net, you can simply execute the following
pipeline-creator (without spending time on thinking how to chain actions in
a workflow):

.. code-block:: python

    from radio.pipelines import get_crops

    pipe = get_crops(fmt='raw', shape=(128, 256, 256),
                     nodules_df=nodules, batch_size=20,
                     share=0.6, nodule_shape=(32, 64, 64))

    (ctset >> pipe).gen_batch(batch_size=12, shuffle=True)

    for batch in gen_batches:
        # ...
        # perform net training here

See the :doc:`documentation <intro/pipelines>` for more information about
ready-made workflows.

Adding a neural-net model to a workflow
---------------------------------------

Contains neural nets' architectures for task of classification,
segmentation and detection. E.g., ``DenseNoduleNet``, an architecutre,
inspired by `DenseNet <https://arxiv.org/abs/1608.06993>`_, but suited for 3D scans.

.. code-block:: python

    from radio.models import DenseNoduleNet

Using the architectures from Models, one can train deep learning systems
for cancer detection. E.g., initialization and training of a new DenseNoduleNet
on scan crops of shape **[32, 64, 64]** can be implemented as follows:

.. code-block:: python

    from radio.preprocessing import CTImagesMaskedBatch as CTIMB
    from dataset import F

    training_flow = (
        ctset
        .pipeline()
        .load(fmt='raw')
        .fetch_nodules_info(nodules_df)
        .create_mask()
        .sample_nodules(nodule_size=(32, 64, 64), batch_size=20)
        .init_model(mode='static', model_class=DenseNoduleNet, model_name='dnod_net')
        .train_model('dnod_net', feed_dict={
            'images': F(CTIMB.unpack, component='images'),
            'labels': F(CTIMB.unpack, component='classification_targets')
        })
    )

    training_flow.run(batch_size=10)


The :doc:`documentation <intro/models>` contains more information about implemented
architectures and their application to cancer detection.

Installation
============

With `pipenv <https://docs.pipenv.org/>`_::

    pipenv install git+https://github.com/analysiscenter/radio.git#egg=radio

With `pip <https://pip.pypa.io/en/stable/>`_::

    pip3 install git+https://github.com/analysiscenter/radio.git


After that just import `RadIO`::

    import radio


.. note:: `RadIO` module is in the beta stage. Your suggestions and improvements are very welcome.

.. note:: `RadIO` supports python 3.5 or higher.

.. note:: When cloning repo from GitHub use flag ``--recursive`` to make sure that ``Dataset`` submodule is also cloned.

    ``git clone --recursive https://github.com/analysiscenter/radio.git``


Citing RadIO
==============
Please cite RadIO in your publications if it helps your research.::

Khudorozhkov R., Emelyanov K., Koryagin A., Ushakov A. RadIO library for data science research of CT images. 2017.
