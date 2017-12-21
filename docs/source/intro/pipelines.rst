
Pipelines
=========

Pipelines are workflows that greatly simplify
deep learning research on CT-scans. Each workflow is represented
in a form of preprocessing `actions https://analysiscenter.github.io/dataset/intro/batch.html#action-methods`_,
chained in a :class:`pipeline <dataset.Pipeline>`.

Let us start with a workflow that allows to perform a full-scale
preprocessing over a dataset of scans and start training the model
of your choice.

Preprocessing workflow
----------------------

Say, you need a workflow that loads scans from disk, resizes them
to shape **[128, 256, 256]**, and prepares batch of **20**
cancerous and non-cancerous crops of shape **[32, 64, 64]**. The straightforward
approach is to chain several actions:

.. code-block:: python

    import pandas as pd
    from radio import dataset as ds

    nodules = pd.read_csv('/path/to/annotation/nodules.csv')

    get_crops = (
      ds.Pipeline()
        .load(fmt='raw')
        .fetch_nodules_info(nodules=nodules)
        .unify_spacing(shape=(128, 256, 256), method='pil-simd',
                       padding='reflect', spacing=(1.7, 1.0, 1.0))
        .create_mask()
        .normalize_hu()
        .sample_nodules(nodule_size=(32, 64, 64), batch_size=20, share=0.5)
        .run(batch_size=8, lazy=True, shuffle=True)
    )

The simpler approach is to use ``get_crops``-function that manufactures frequently
used preprocessing pipelines. With ``get_crops`` you can get the pipeline written above
in two lines of code:

.. code-block:: python

    from radio.pipelines import get_crops
    pipeline = get_crops(fmt='raw', shape=(128, 256, 256), nodules=nodules, histo=some_3d_histogram,
                         batch_size=20, share=0.6, nodule_shape=(32, 64, 64))

Pay attention to parameters ``batch_size`` and ``share``: they allow
to control the number of items in a batch of crops and the number
of cancerous crops. Parameter ``histo`` controls the distribution which
is used for sampling locations of random (noncancerous) crops. Although
``histo`` accepts any 3d-histogram, we advise to use
`histogram based on distribution of cancer location  <Calculation of cancer location distribution>`_.

You can chain ``pipeline`` with some additional actions for training, say, ``DenseNoduleNet``:

.. code-block:: python

    pipeline = (
        pipeline
        .init_model('static',class=DenseNoduleNet, model_name='dnod_net')
        .train_model(model_name='dnod_net', feed_dict={
            'images': F(CT.unpack, component='images'),
            'labels': F(CT.unpack, component='classification_targets')
        })
    )
    (ctset >> pipeline).run()

Alternatively, you can choose to save dataset of crops
on disk and get back to training a network on it later:

.. code-block:: python

    pipeline = pipeline.dump('/path/to/crops/')
    (ctset >> pipeline).run()

Created pipeline will generate `~1500`
training examples, in one run through Luna-dataset
(one epoch). It may take a couple of hours to
work through the pipeline, even for a high performing machine.
The reason for this is that both ``resize`` and ``load`` are costly
operations.

That being said, for implementing an efficient learning procedure
we advise to use another workflow, that allows to generate more
than `100000` training examples after running one time through
the Luna-dataset.

**Requirements** for ``get_crops``: Dataset of scans in **DICOM** or **MetaImage**. ``pandas.DataFrame``
of nodules-annotations in `Luna-format <https://luna16.grand-challenge.org/data/>`_.

Faster workflow
---------------

Preparation of richer training dataset can be achieved in two steps using two pipeline-creators:
``split_dump`` and ``combine_crops``.

**Step 1**

During the first step you dump large sets of cancerous and non-cancerous
crops in separate folders using ``split_dump``:

.. code-block:: python

    from radio.pipelines import split_dump
    pipeline = split_dump(cancer_path='/train/cancer', non_cancer_path='/train/non_cancer',
                          nodules=nodules)
    (ctset >> pipeline).run()  # one run through Luna; may take a couple of hours

**Requirements for** ``split_dump``: Dataset of scans in **DICOM** or **MetaImage**. ``pandas.DataFrame``
    of nodules-annotations in `Luna-format <https://luna16.grand-challenge.org/data/>`_.

**Step 2**

You can now combine cancerous and non-cancerous crops from two folders using ``combine_crops``.
First, you associate a :class:`dataset <dataset.Dataset>` with each folder:

.. code-block:: python

    # datasets of cancerous and non-cancerous crops
    cancer_set = Dataset(index=FilesIndex('/train/cancer/*', dirs=True))
    non_cancer_set = Dataset(index=FilesIndex('/train/non_cancer/*', dirs=True))

You can balance crops from two dataset in any proportion you want:

.. code-block:: python

    from radio.pipelines import combine_crops
    pipeline = combine_crops(cancer_set, non_cancer_set, batch_sizes=(10, 10))

Pay attention to parameter ``batch_sizes`` in ``combine_crops``-functions.
It defines how many cancerous and non-cancerous crops will be included
in batches. Just like with `get_crops`, it is easy to add training of *ResNet* to
``pipeline``:

.. code-block:: python

    pipeline = (
        pipeline
        .init_model('static',class=ResNodule3DNet50, model_name='resnet')
        .train_model(model_name='resnet', feed_dict={
        'images': F(CT.unpack, component='images'),
            'labels': F(CT.unpack, component='classification_targets')
        })
    )
    (ctset >> pipeline).run(BATCH_SIZE=12)

*\ **Requirements**\ * for ``combine_crops``: datasets of cancerous and noncancerous crops, prepared
by ``split_dump`` (see  **Step 1** ).

Calculation of cancer location distribution
-------------------------------------------
Another useful pipeline-creator is ``update_histo``. With ``update_histo`` you can get a histogram-estimate
of distribution of cancer-location inside preprocessed scans:

.. code-block:: python

    from radio.pipelines import update_histo
    SHAPE = (400, 512, 512)  # default shape of resize in preprocessing
    ranges = list(zip([0]*3, SHAPE)) # boxes of preprocessed scans
    histo = list(np.histogramdd(np.empty((0, 3)), range=ranges, bins=4))  # init empty 3d-histogram

    pipeline = update_histo(nodules, histo)

It is time to run a dataset of scans through ``pipeline`` and accumulate information about cancer-location
in ``histo``:

.. code-block:: python

    (ctset >> pipeline).run() # may take a couple of hours

You can now use ``histo`` in pipeline ``get_crops`` to sample batches of cancerous and noncancerous crops:

.. code-block:: python

        pipeline = get_crops(nodules=nodules, histo=histo)

In that way, cancerous and noncancerous examples will be cropped from similar locations. This, of course, makes
training datasets more balanced.

**Requirements** for ``get_crops``: Dataset of scans in **DICOM** or **MetaImage**. ``pandas.DataFrame``
of nodules-annotations in `Luna-format <https://luna16.grand-challenge.org/data/>`_.
