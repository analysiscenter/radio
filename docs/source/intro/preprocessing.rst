
Preprocessing
=============

Module allows to perform a number of preprocessing actions on a dataset of scans.

`RadIO` works with batches of scans, wrapped in a class ``CTImagesBatch``.
The class stores scans' data in :class:`components <dataset.Batch>` which represents main attributes
of scans. E.g., scans itself are stacked in one
tall 3d ``numpy``-array (called `skyscraper`) and stored in ``images``-component. The other
components are ``spacing`` and ``origin``, which store important meta.

So, what can be done with the data?

Load and dump
-------------

The ``preprocessing``-module is primarily adapted to work with two
large datasets, available in open access: `Luna-dataset <https://luna16.grand-challenge.org/data/>`_
(**.raw**-format) and `DsBowl2017-dataset <https://www.kaggle.com/c/data-science-bowl-2017>`_ (**.dicom**-format).
Consider, you have one of these two datasets (or a part of it) downloaded in
folder `path/to/scans`. The first step is to set up an index.
:class:`~dataset.FilesIndex` from :mod:`dataset`-module reduces the
task to defining a ``glob``-mask for a needed set of scans:

.. code-block:: python

    from radio import CTImagesBatch
    from radio.dataset import FilesIndex, Dataset, Pipeline
    ctset = FilesIndex(path='path/to/scans/*', no_ext=True)
    pipeline = Pipeline()

For loading scans you only need to call action ``load`` specifying
format of dataset:

.. code-block:: python

    pipeline = pipeline.load(fmt='raw') # use fmt = 'dicom' for load of dicom-scans

After performing some preprocessing operations you may need to save the
results on disk. Action ``dump`` is of help here:

.. code-block:: python

    pipeline = ... # some preprocessing actions
    pipeline = pipeline.dump(dst='path/to/preprocessed/')

In the end, data of each scan from the batch will be packed with
:func:`blosc <blosc.pack_array>` and dumped into folder.
Dumped scans can be loaded later using the same methodology.
To do this, specify `blosc`-format when performing ``load``:

.. code-block:: python

    pipeline = Pipeline().load(fmt='blosc')

Both ``dump`` and ``load`` from `blosc` can work component-wise:

.. code-block:: python

    pipeline_dump = (
        Pipeline()
        .dump(fmt='blosc', src=['spacing', 'origin']) # dump spacing, origin components
        .dump(dst='path/to/preprocessed/', fmt='blosc', src='images') # dumps scans itself
    )

    pipeline_load = Pipeline().load(fmt='blosc', src_blosc=['spacing', 'origin', 'images']) # equivalent to src_blosc=None


Resize and unify spacing
------------------------

Another step of preprocessing is **resize** of scans to a specific shape.
``preprocessing``-module has ``resize``-action, specifying desired
output shape in z, y, x order:

.. code-block:: python

    batch = batch.resize(shape=(128, 256, 256))

Currently module supports two different resize-engines:
:mod:`scipy.interpolate` and ``PIL-simd``. While the second engine
is more robust and works faster on systems with small number
of cores, the first allows greater degree of parallelization
and can be more precise in some cases. One can choose engine
in a following way:

.. code-block:: python

    batch = batch.resize(shape=(128, 256, 256), method='scipy')

Sometimes, it may be useful to convert scans to the same real-world scale,
rather than simply reshape to same size. It might be useful if parts of scans
with similar real-world shapes would have same voxel-sizes.
This can be achieved through ``unify_spacing``-action:

.. code-block:: python

    batch = batch.unify_spacing(spacing=(3.0, 2.0, 2.0), shape=(128, 256, 256))

To control real-world world scale of scans, you can specify ``spacing``,
that represents distances in millimeters between adjacent voxels along three axes.
The action works in two steps. The first step stands for spacing
unification by means of resize, while the second one crops/pads
resized scan so that it fits in the supplied shape. You can specify
resize parameters and padding mode:

.. code-block:: python

    batch = batch.unify_spacing(spacing=(3.0, 2.0, 2.0), shape=(128, 256, 256),
                                padding='reflect', engine='pil-simd')

So far it was all about ``images``-components, that can be viewed as
an **X**-input of a net. What about net's target, **Y**-input?

Create masks with ``CTImagesMaskedBatch``
-----------------------------------------

Preparing target for network revolves around class ``CTImagesMaskedBatch``.
It naturally has one new component - ``masks``. ``Masks`` have the same
shape as ``images`` and store cancer-masks of different items
in a binary format, where value of each voxel is either **0** (non-cancerous voxel) or
**1** (cancerous voxel). ``masks`` can be made in two steps.
First, load info about cancerous nodules in a batch:

.. code-block:: python

    pipeline = (
        Pipeline()
         .fetch_nodules_info(nodules_df=nodules_df) # nodules_df is a Pandas.DataFrame
                                                    # containing info about nodules
    )

Then you can fill the ``masks``-component using the loaded info:

.. code-block:: python

    pipeline = (
        pipeline
        .create_mask()
    )

Sample crops from scan: preparing training examples for neural net
--------------------------------------------------------------------

RadIO has ``sample_nodules`` that allows to generate batches of small crops, balancing cancerous
and non-cancerous examples.
Let's start preprocessing with ``resize`` of scans:

.. code-block:: python

    pipeline = (
        pipeline
        .resize(shape=(256, 512, 512))
    )

Now all scans have the same shape **(256, 512, 512)**, it is
possible to put them into a neural net. However, it may fail for two main reasons:

* only small number of scans (say, 3) of such size can be put into a memory of a GPU
* typically, there are not so many scans available for training (888 for Luna-dataset). As a result, making only one training example out of a scan is rather wasteful.

A more efficient approach is to crop out interesting parts of scans. E.g., this
piece of code

.. code-block:: python

    pipeline = (
        pipeline
        .resize(shape=(256, 512, 512))
        .sample_nodules(nodule_size=(32, 64, 64),
                        batch_size=20, share=0.5)
    )

will generate batches of size **20**, that will contain **10** cancerous and **10**
noncancerous crops of shape **(32, 64, 64)**. Or, alternatively this code

.. code-block:: python

    pipeline = (
        pipeline
        .resize(shape=(256, 512, 512))
        .sample_nodules(nodule_size=(32, 64, 64),
                        batch_size=20, share=0.6,
                        variance=(100, 200, 200),
                        histo=some_3d_histogram)
    )

will generate batches of size **20** with **12** cancerous crops. Pay attention to
parameters ``variance`` and ``histo``:

* ``variance`` introduces variability in the location of cancerous nodule inside the crop. E.g., if set to **(100, 200, 200)**, the location of cancerous nodule will be sampled from normal distribution with zero-mean and variances **(100, 200, 200)** along three axes.

* ``histo`` allows you to control the positions of noncancerous crops. If ``histo`` set to ``None``, noncancerous crops will be sampled uniformly from scan-boxes of shape **(256, 512, 512)**. Sometimes, though, you may want to sample noncancerous crops from specific regions of lungs - say, the interior of the left lung. In this case you can generate a 3d-histogram (see :func:`numpy.histogram`) concentrated in this region and supply it into ``sample_nodules``-action.


Augment data on-the-fly
-----------------------

Medical datasets are often small and require additional augmentation to avoid overfitting. For this purpose, it is possible to combine ``rotate`` and ``central_crop``:

.. code-block:: python

    pipeline = (
        pipeline
        .resize(shape=(256, 512, 512))
        .rotate(angle=90, axes=(1, 2), random=True)
        .central_crop(crop_size=(32, 64, 64))
    )

This pipeline first resize all images to same shape and then sample rotated crops of shape **[32, 64, 64]**,
rotation angle is random, from 0 to 90 degrees. Rotation is performed along **y and x** axes.
Crops are padded by zeroes after rotation, if needed.

Accessing Batch components
--------------------------

You may want to access ``CTImagesBatch`` or ``CTImagesMaskedBatch``-data directly. E.g., if you decide to write your own :func:`actions <dataset.action>`.
Batch-classes has such functionality: 3d-scan for an item indexed by ``nb``
from a ``batch`` can be accessed in the following way:

.. code-block:: python

    image_3d_nb = batch.get(nb, 'images')

The same goes for other components of item ``nb``:

.. code-block:: python

    spacing_nb = batch.get(nb, 'spacing')
