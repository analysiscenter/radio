
Dataset
=======

Creating a dataset
------------------

The `Dataset` holds an index of all data items (e.g. customers, transactions, etc) and a specific action class
to process a small subset of data (batch).

.. code-block:: python

   from dataset import DatasetIndex, Dataset, Batch

   client_index = DatasetIndex(client_ids_list)
   client_dataset = Dataset(client_index, batch_class=Batch)


Preloading data
---------------
For smaller dataset it might be convenient to preload all data at once:

.. code-block:: python

   client_dataset = Dataset(client_index, batch_class=Batch, preloaded=some_data)

As a result, all created batches will contain a portion of `some_data`.


Splitting a dataset
-------------------
A dataset can be easily split into train, test and validation subsets.

.. code-block:: python

    client_dataset.cv_split([.8, .1, .1])

All parts are also datasets, which can be addressed as `dataset.train`, `dataset.test` and `dataset.validation`.

Parameters
^^^^^^^^^^
`shares` - train/test/validation shares. Can be float, tuple of 2 floats, or tuple of 3 floats.

`shuffle` - whether to randomize items order before splitting into subsets. Default is `False`. Can be

* `bool` : `False` - to make subsets in the order of indices in the index, or `True` - to make random subsets.
* a :class:`numpy.random.RandomState` object which has an inplace shuffle method.
* `int` - a random seed number which will be used internally to create a :class:`numpy.random.RandomState` object.
* `callable` - a function which gets an order and returns a shuffled order.


Iterating over a dataset
------------------------

And now you can conveniently iterate over the dataset:

.. code-block:: python

   BATCH_SIZE = 200
   for client_batch in client_dataset.gen_batch(BATCH_SIZE, shuffle=False, n_epochs=1):
       # client_batch is an instance of DataFrameBatch which holds an index of the subset of the original dataset
       # so you can do anything you want with that batch
       # for instance, load some data, as the batch is empty when initialized
       batch_with_data = client_batch.load(client_data)

You might also create batches with `next_batch` function:

.. code-block:: python

   NUM_ITERS = 1000
   for i in range(NUM_ITERS):
       client_batch = client_dataset.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
       batch_with_data = client_batch.load(client_data)
       # ...

The only difference is that :func:`~dataset.Dataset.gen_batch` is a generator,
while :func:`~dataset.Dataset.next_batch` is just an ordinary method.

Parameters
^^^^^^^^^^
`batch_size` - number of items in the batch.

`shuffle` - whether to randomize items order before splitting into batches. Default is `False`. Can be

* `bool` : `False` - to make batches in the order of indices in the index, or `True` - to make random batches.
* a :class:`numpy.random.RandomState` object which has an inplace shuffle method.
* `int` - a random seed number which will be used internally to create a :class:`numpy.random.RandomState` object.
* `sample function` - any callable which gets an order and returns a shuffled order.

`n_epochs` - number of iterations around the whole dataset. If `None`\ , then you will get an infinite sequence of batches. Default value - 1.

`drop_last` - whether to skip the last batch if it has fewer items (for instance, if a dataset contains 10 items and the batch size is 3, then there will 3 batches of 3 items and the 4th batch with just 1 item. The last batch will be skipped if `drop_last=True`).


Custom batch class
------------------
You can also define a new :doc:`batch class <batch>` with custom action methods to process your specific data.

.. code-block:: python

    class MyBatch(Batch):
        @action
        def my_custom_action(self):
            ...

        @action
        def another_custom_action(self):
            ...

And then create a dataset with a new batch class:

.. code-block:: python

   client_dataset = Dataset(client_index, batch_class=MyBatch)

API
---

See :doc:`Dataset API <../api/dataset.dataset>`.
