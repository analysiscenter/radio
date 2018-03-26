
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
folder ``path/to/scans``. The first step is to set up an index.
:class:`~dataset.FilesIndex` from :mod:`dataset`-module reduces the
task to defining a ``glob``-mask for a needed set of scans:

.. code-block:: python

    from radio import CTImagesBatch as CTIMB
    from radio.dataset import FilesIndex, Dataset, Pipeline

    ctx = FilesIndex(path='path/to/scans/*', no_ext=True)
    ctset = Datset(index=ctx, batch_class=CTIMB)
    pipeline = Pipeline()

For loading scans you only need to call action  :meth:`~.CTImagesBatch.load` specifying
format of dataset:

.. code-block:: python

    pipeline = pipeline.load(fmt='raw') # use fmt = 'dicom' for load of dicom-scans

After performing some preprocessing operations you may need to save the
results on disk. Action :meth:`~.CTImagesBatch.dump` is of help here:

.. code-block:: python

    pipeline = ... # some preprocessing actions
    pipeline = pipeline.dump(dst='path/to/preprocessed/')

In the end, data of each scan from the batch will be packed with
:func:`blosc <blosc.pack_array>` and dumped into folder.
Dumped scans can be loaded later using the same methodology.
To do this, specify `blosc`-format when performing :meth:`~.CTImagesBatch.load`:

.. code-block:: python

    pipeline = Pipeline().load(fmt='blosc')

Both :meth:`~.CTImagesBatch.dump` and :meth:`~.CTImagesBatch.load` from `blosc` can work component-wise:

.. code-block:: python

    pipeline_dump = (
        pipeline
        .dump(fmt='blosc', components=['spacing', 'origin']) # dump spacing, origin components
        .dump(dst='path/to/preprocessed/', fmt='blosc', components='images') # dumps scans itself
    )

    pipeline_load = Pipeline().load(fmt='blosc', components=['spacing', 'origin', 'images'])


.. _ResizeUspac:

Resize and unify spacing
------------------------

Another step of preprocessing is **resize** of scans to a specific shape.
``preprocessing``-module has :meth:`~.CTImagesBatch.resize`-action, specifying desired
output shape in z, y, x order:

.. code-block:: python

    pipeline = pipeline.resize(shape=(128, 256, 256))

Currently module supports two different resize-engines:
:mod:`scipy.interpolate` and ``PIL-simd``. While the second engine
is more robust and works faster on systems with small number
of cores, the first allows greater degree of parallelization
and can be more precise in some cases. One can choose engine
in a following way:

.. code-block:: python

    pipeline = pipeline.resize(shape=(128, 256, 256), method='scipy')

Sometimes, it may be useful to convert scans to the same real-world scale,
rather than simply reshape to same size.
This can be achieved through :meth:`~.CTImagesBatch.unify_spacing`-action:

.. code-block:: python

    pipeline = pipeline.unify_spacing(spacing=(3.0, 2.0, 2.0), shape=(128, 256, 256))

To control real-world world scale of scans, you can specify ``spacing``,
that represents distances in millimeters between adjacent voxels along three axes.
The action works in two steps. The first step stands for spacing
unification by means of resize, while the second one crops/pads
resized scan so that it fits in the supplied shape. You can specify
resize parameters and padding mode:

.. code-block:: python

    pipeline = pipeline.unify_spacing(spacing=(3.0, 2.0, 2.0), shape=(128, 256, 256),
                                padding='reflect', engine='pil-simd')

So far it was all about ``images``-components, that can be viewed as
an **X**-input of a neural network. What about network's target, **Y**-input?


Create masks with ``CTImagesMaskedBatch``
-----------------------------------------

Preparing target for network revolves around class ``CTImagesMaskedBatch``.
It naturally has one new component - ``masks``. ``Masks`` have the same
shape as ``images`` and store cancer-masks of different items
in a binary format, where value of each voxel is either **0** (non-cancerous voxel) or
**1** (cancerous voxel). ``masks`` can be made in two steps.
First, load info about cancerous nodules in a batch with :meth:`~.CTImagesMaskedBatch.fetch_nodules_info`:

.. code-block:: python

    pipeline = (
        pipeline
         .fetch_nodules_info(nodules=nodules) # nodules is a Pandas.DataFrame
                                              # containing info about nodules
    )

Then you can fill the ``masks``-component using the loaded info and action :meth:`~.CTImagesMaskedBatch.create_mask`:

.. code-block:: python

    pipeline = (
        pipeline
        .create_mask()
    )


.. _sample_crops_from_scan:

Sample crops from scan
----------------------

RadIO has :meth:`~.CTImagesMaskedBatch.sample_nodules` that allows to generate batches of small crops, balancing cancerous
and non-cancerous examples.
Let's start preprocessing with :ref:`resize <ResizeUspac>` of scans:

.. code-block:: python

    pipeline = (
        pipeline
        .resize(shape=(256, 512, 512))
    )

Now all scans have the same shape **(256, 512, 512)**, it is
possible to put them into a neural network. However, it may fail for two main reasons:

* only small number of scans (say, 3) of such size can be put into a memory of a GPU
* typically, there are not so many scans available for training (888 for Luna-dataset). As a result, making only one training example out of a scan is rather wasteful.

A more efficient approach is to crop out interesting parts of scans using :meth:`~.CTImagesMaskedBatch.sample_nodules`.
E.g., this piece of code

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
parameters ``variance`` and ``histo`` of :meth:`~.CTImagesMaskedBatch.sample_nodules`:

* ``variance`` introduces variability in the location of cancerous nodule inside the crop. E.g., if set to **(100, 200, 200)**, the location of cancerous nodule will be sampled from normal distribution with zero-mean and variances **(100, 200, 200)** along three axes.

* ``histo`` allows you to control the positions of noncancerous crops. If ``histo`` set to ``None``, noncancerous crops will be sampled uniformly from scan-boxes of shape **(256, 512, 512)**. Sometimes, though, you may want to sample noncancerous crops from specific regions of lungs - say, the interior of the left lung. In this case you can generate a 3d-histogram (see :func:`numpy.histogram`) concentrated in this region and supply it into ``sample_nodules``-action.


Augment data on-the-fly
-----------------------

Medical datasets are often small and require additional augmentation to avoid overfitting. For this purpose, it is possible to combine
:meth:`~.CTImagesBatch.rotate` and :meth:`~.CTImagesMaskedBatch.central_crop`:

.. code-block:: python

    pipeline = (
        pipeline
        .resize(shape=(256, 512, 512))
        .rotate(angle=90, axes=(1, 2), random=True)
        .central_crop(crop_size=(32, 64, 64))
    )

This pipeline first resizes all images to same shape and then samples rotated crops of shape **[32, 64, 64]**;
rotation angle is random, from 0 to 90 degrees. Rotation is performed about **z**-axis.
Crops are padded by zeroes after rotation, if needed.


Accessing Batch components
--------------------------

You may want to access ``CTImagesBatch`` or ``CTImagesMaskedBatch``-data directly. E.g., if you decide to write your own :func:`actions <dataset.action>`.
Batch-classes has such functionality: 3d-scan for an item indexed by ``ix``
from a ``batch`` can be accessed in the following way:

.. code-block:: python

    image_3d_ix = batch.get(ix, 'images')

The same goes for other components of item ``ix``:

.. code-block:: python

    spacing_ix = batch.get(ix, 'spacing')

Or, alternatively

.. code-block:: python

    image_3d_ix = getattr(batch[ix], 'images')
    spacing_ix = batch[ix].spacing

It is sometimes useful to print indices of all items from a ``batch``:

.. code-block:: python

    print(batch.indices) # batch.indices is a list of indices of all items


Writing your own actions
------------------------

Now that you know how to work with components of :class:`~.CTImagesBatch`, you can write your own action. E.g., you need an
action, that subtracts mean-values of voxel densities from each scan. You can easily inherit one of
batch classes of **RadIO** (we suggest to use :class:`~.CTImagesMaskedBatch`) add make your action ``center`` a method of this
class, just like that:

.. code-block:: python

    from RadIO.dataset import action
    from RadIO import CTImagesMaskedBatch

    class CTImagesCustomBatch(CTImagesMaskedBatch):
        """ Ct-scans batch class with your own action """

        @action  # action-decorator allows you to chain your method with other actions in pipelines
        def center(self):
            """ Center values of pixels in each scan from batch """
            for ix in self.indices:
                mean_ix = np.mean(self.get(ix, 'images'))
                images_ix = getattr(self[ix], 'images')
                images_ix[:] -= mean_ix

            return self  # action must always return a batch-object

You can then chain your action ``center`` with other actions of :class:`~.CTImagesMaskedBatch`
to form custom preprocessing pipelines:

.. code-block:: python

    pipeline = (Pipeline()
                .load(fmt='blosc')  # load data
                .center()  # mean-normalize scans
                .sample_nodules(batch_size=20))  # sample cancerous and noncancerous crops
