
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

The workflow, that includes load of data from disk, resize
to shape **[128, 256, 256]**, and preparing batch of **20**
cancerous and non-cancerous crops of shape **[32, 64, 64]**,
can be set up in a following way:

.. code-block:: python

    from radio.pipelines import get_crops
    pipeline = get_crops(fmt='raw', shape=(128, 256, 256), nodules_df=nodules,
                                batch_size=20, share=0.6, nodule_shape=(32, 64, 64))

Pay attention to parameters ``batch_size`` and ``share``: they allow
to control the number of items in a batch of crops and the number
of cancerous crops. You can chain ``pipeline`` with some additional actions
for training, say, ``DenseNoduleNet``:

.. code-block:: python

    pipeline = (
        pipeline
        .init_model('static',class=DenseNoduleNet, model_name='dnod_net')
        .train_model(model_name='dnod_net', feed_dict={
            'images': F(CT.unpack, component='images'),
            'labels': F(CT.unpack, component='classification_targets')
        })
    )
    (ctset >> pipeline).run(BATCH_SIZE=12)

Alternatively, you can choose to save dataset of crops
on disk and get back to training a network on it later:

.. code-block:: python

    pipeline = pipeline.dump('/path/to/crops/')
    (ctset >> pipeline).run(BATCH_SIZE=12)

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

Faster workflow
---------------

Preparation of richer training dataset can be achieved in two steps.
During the first step you dump large sets of cancerous and non-cancerous
crops in separate folders:

.. code-block:: python

    from radio.pipelines import split_dump
    pipeline = split_dump(cancer_path='/train/cancer', non_cancer_path='/train/non_cancer')
    (ctset >> pipeline).run()

You can combine cancerous and non-cancerous crops from two folders.
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
in batches.
