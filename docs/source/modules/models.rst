Models
======

RadIO package contains implementation of several neural network architectures
that can be used for solving lung cancer detection problem.

There three main tasks that can be solved by lung cancer detection systems:

- segmentation
- classification
- regression

Each of these categories requires specific ANN architecture, loss function
and target values:

+----------------+---------------+----------------------------+
|      Task      |     Loss      |      Target shape          |
+================+===============+============================+
| Classification |   log loss    |     (batch_size, 1)        |
+----------------+---------------+----------------------------+
| Regression     |  custom loss  |    (batch_size, 7)         |
+----------------+---------------+----------------------------+
|  Segmentation  | dice,tiversky | (batch_size, *scan_shape)  |
+----------------+---------------+----------------------------+

Description of implemented loss functions:

- :doc:`tensorflow losses <../api/tensorflow_loss>`
- :doc:`keras losses <../api/keras_loss>`

------------------------------------------------------------------------------------

* In case of segmentation CNN is trained to predict binary mask.
  Each pixel of the mask may be 1 for cancerous region or 0 otherwise.

* In case of regression task CNN is trained to predict location, sizes and probability
  of cancer tumor at once. Pulling all together, there is 7 target values:
  three for location, three for sizes and one for probability of cancer.


* In case of classification task of lung-cancer detection is just a labeling input
  crops or scans with 1 for cancerous scans and 0 for non-cancerous.

------------------------------------------------------------------------------------


Full list of ANN models contained in RadIO.models submodule:

+---------------+----------------+-------------+--------------+
|     Model     | Classification |  Regression | Segmentation |
+===============+================+=============+==============+
| KerasResNet50 |        \+      |      \+     |       \+     |
+---------------+----------------+-------------+--------------+
| KerasVGG16    |        \+      |      \+     |       \-     |
+---------------+----------------+-------------+--------------+
| KerasVnet     |        \-      |      \-     |       \+     |
+---------------+----------------+-------------+--------------+
| TFResNet      |        \+      |      \+     |       \-     |
+---------------+----------------+-------------+--------------+
| TFDilatedVnet |        \-      |      \-     |       \+     |
+---------------+----------------+-------------+--------------+
| TFDenseNet    |        \+      |      \+     |       \-     |
+---------------+----------------+-------------+--------------+


Models interface is implemented without any binding to CTImagesBatch
and CTImagesMaskedBatch structure. Typically, models accepts input data in
tuple of ndarray's or dict with values being ndarrays. Each type of task
requires specific function that unpacks data from CTImagesBatch into
suitable for model learning or prediction format.
RadIO contains three functions for unpacking data from batch:

- :doc:`unpack_clf <../api/unpack_clf>` for classification task
- :doc:`unpack_reg <../api/unpack_reg>` for regression task
- :doc:`unpack_seg <../api/unpack_seg>` for segmentation task


All models can be trained directly in
pipeline run-loop using dataset.Pipeline methods. For example, following code
trains TFResNet model on classification task:

.. code-block:: python

  from lung_cancer.models.tensorflow import TFResNet
  from lung_cancer import dataset as ds

  resnet_config = {
    'num_targets': 1,
    'optimizer': 'Adam',
    'loss': tf.losses.log_loss,
  }

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', TFResNet, 'resnet', config=resnet_config)
      .train_model('resnet', unpack_clf)
  )

Now train loop can be started:

.. code-block:: python

  (train_dataset >> train_ppl).run(batch_size=16)

Pipeline instances contain several special methods that can be built in chain of
batch action-decorated methods. In example above `init_model` and `train_model`
methods are methods of ds.Pipeline instances.

**init_model** method is called just once
when pipeline object is being constructed. First argument of this method is
type of model: 'static' or 'dynamic'. Second -- model's class,
third argument -- name of model, and last one -- model's configuration dict.
Configuration dictionary may contain parameters that will be used by a model
when it is being built. More information about configuration dictionary, models types
and their interaction with ds.Pipeline instances
can be found in `models section <https://analysiscenter.github.io/dataset/intro/models.html>`_
of dataset package documentation.

**train_model** method accepts name of the model as its first argument and
callable that can be used for unpacking data from batch in a format suitable for
ANN learning. This method is called on every iteration.

Full description `ds.Pipeline`'s methods that enables interaction with models
can be in `dataset <https://analysiscenter.github.io/dataset/intro/models.html>`_ package documentation.

The same ResNet model can be configured for regression task: the only thing
required is to change number of target values and loss functions
in configuration dictionary. Also, another function for unpacking data from
CTImagesMaskedBatch will be used:

.. code-block:: python

  from lung_cancer import dataset as ds
  from lung_cancer.models import unpack_reg
  from lung_cancer.models.tensorflow import TFResNet, reg_l2_loss

  resnet_config = {
    'num_targets': 7,
    'optimizer': 'Adam',
    'loss': reg_l2_loss
  }

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', TFResNet, 'resnet', config=resnet_config)
      .train_model('resnet', unpack_reg)
  )

Training segmentation CNN is as simple as training regression or classification
models:

.. code-block:: python

  from functools import partial
  from lung_cancer import dataset as ds
  from lung_cancer.models import unpack_seg
  from lung_cancer.models.keras import KerasVnet
  from lung_cancer.models.keras.losses import dice_loss, tiversky_loss

  vnet_config = {
    'optimizer': 'Adam',
    'loss': tiversky_loss
  }

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', KerasVnet, 'vnet', config=vnet_config)
      # KerasVnet has 'channels_first' dim_ordering, but
      # unpack_seg has 'channels_last' by default, so we need partial from functools
      .train_model('vnet', partial(unpack_seg, dim_ordering='channels_first'))
  )


Adding metrics computation into training loop can be performed as:

.. code-block:: python

  from lung_cancer.models.metrics import accuracy, recall, precision, log_loss

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', TFResNet, 'resnet', config=resnet_config)
      .train_model('resnet', unpack_reg)
  )
