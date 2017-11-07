.. RadIO documentation master file, created by
   sphinx-quickstart on Mon Nov  6 23:59:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RadIO's documentation!
=================================

`RadIO` is a framework for processing of batches of computational
tomography (CT)-scans, with purpose of detecting cancerous nodules.
The framework incorporates several modules:

* preprocessing
* models
* pipelines

Contents
========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   modules/preprocessing
   modules/pipelines
   modules/models
   api/api


Preprocessing
-------------

This module contains implementation of a set of `actions
<https://github.com/analysiscenter/dataset>`_, that allow to
efficiently prepare a dataset of CT-scans for analysis with
the use of neural nets. Say, we trained a net that allows
to classify CT-scans of lungs of shape **[128, 256, 256]**
as having cancerous nodules or not. We also need to classify
a dataset of scans of shape **[256, 512, 512]** stored in
**DICOM** format. Moreover, the net is trained on scans
with pixel densities normalized to range **[0, 255]**. Clearly,
if we want to obtain sensible results, we need to preprocess
the scans in a similar way before putting them into the net.
Preprocessing and dump of a set of scans can be easily done
with the use of the following `pipeline
<https://github.com/analysiscenter/dataset>`_, just like that:

.. code-block:: python

    from lung_cancer import CTImagesBatch
    from lung_cancer.dataset import FilesIndex, Dataset
    dicomix = FilesIndex(path='path/to/dicom/*', no_ext=True) # set up the index
    dicomset = Dataset(index=dicomix, batch_class=CTImagesBatch) # init the dataset of blosc files
    dir_dump = '/path/to/preprocessed/' # preprocessed scans are stored here
    prep_ppl = (dicomset.pipeline() # set up preprocessing workflow
                .load(fmt='dicom')
                .resize(shape=(128, 256, 256))
                .normalize_hu()
                .dump(dir_dump)) # dump results of the preprocessing

    prep_ppl.run(batch_size=20) # run it only now

See the :doc:`documentation <modules/preprocessing>` for the description of
preprocessing actions implemented in the module.

Pipelines
---------

This module contains helper-functions that generate useful pipelines.
In essence, pipelines represent ready-to-use workflows of scans' data
processing. E.g., if one wants to experiment with training of
[Vnet](linkvnet) on dataset of luna-scans `ctset`, he can simply
execute the following pipeline-creator (without spending time on
thinking how to chain actions in a preprocessing workflow):

.. code-block:: python

    from lung_cancer import get_preprocessed
    pipe = get_preprocessed(fmt='raw', shape=(128, 256, 256), nodules_df=nodules, batch_size=20,
                            share=0.6, nodule_shape=(32, 64, 64))
    pipe = pipe.train_model(model='vnet', model_class=VnetModel)
    (pipe >> ctset).run(BATCH_SIZE=12)

See the :doc:`documentation <modules/pipelines>` for more information about
implemented workflows.

Working with neural nets
------------------------

The module contains neural nets' architectures suitable for
tasks of classification, segmentation, detection. Importantly,
the module incorporates structures, that explicitly show how
to apply the nets for task at hand (cancer detection). Both
training and inferencing are covered. The list of models
implemented in TensorFlow/Keras contains a set of high-performing
architectures, including, but not limited to [VGG](vgglink),
[ResNet](resnetlink) and [Vnet](vnetlink). Initialization and
training of a new model (say, [DenseNet](linkondense)) on scan
crops of shape **[32, 64, 64]** can be implemented as follows:

.. code-block:: python

    from lung_cancer import CTImagesModels, DenseNet
    from lung_cancer.dataset import FilesIndex, Dataset
    prepdir = 'path/to/preprocessed/*' # preprocessed scans are stored here
    prepix = FilesIndex(path=prepdir, dirs=True)
    prepset = Dataset(index=prepix, batch_class=CTImagesModels)
    net = DenseNet('dense_net')
    train_ppl = (prepset.pipeline().
                .load(fmt='blosc')
                .sample_nodules(nodule_size=(32, 64, 64), batch_size=20) # sample 20 crops from scans
                .train_on_crops('dense_net'))

    train_ppl.run(batch_size=10)


The [documentation](linkmodels) contains more information about implemented models and their application to cancer detection.
