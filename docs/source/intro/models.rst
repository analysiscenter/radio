Models
======

RadIO has implementations of neural network architectures
that can be used for lung cancer detection.
Models interface is implemented without any binding to `CTImagesBatch`
and `CTImagesMaskedBatch` structure.
Typically, models accepts input data in tuple of ndarray's or dict
with values being ndarrays.

In RadIO models can be distinguished by tasks that they perform:

* segmentation
* classification
* regression

Each of these categories has notable architectures, loss functions
and target forms.

Architectures
---------------

Full list of models contained in RadIO.models submodule:

+---------------------+----------------+-------------+--------------+
|        Model        | Classification |  Regression | Segmentation |
+=====================+================+=============+==============+
| KerasResNoduleNet50 |        \+      |      \+     |       \-     |
+---------------------+----------------+-------------+--------------+
| KerasNoduleVGG      |        \+      |      \+     |       \-     |
+---------------------+----------------+-------------+--------------+
| Keras3DUNet         |        \-      |      \-     |       \+     |
+---------------------+----------------+-------------+--------------+
| ResNodule3DNet50    |        \+      |      \+     |       \-     |
+---------------------+----------------+-------------+--------------+
| DilatedNoduleNet    |        \-      |      \-     |       \+     |
+---------------------+----------------+-------------+--------------+
| DenseNoduleNet      |        \+      |      \+     |       \-     |
+---------------------+----------------+-------------+--------------+

Loss functions
---------------

Description of implemented loss functions:

- :doc:`tensorflow losses <../api/tf_loss>`
- :doc:`keras losses <../api/keras_loss>`

Network's targets
-----------------

Each type of task requires specific function that unpacks data from ``CTImagesBatch``
or ``CTImagesMaskedBatch`` into suitable format.

------------------------------------------------------------------------------------

* **Segmentation**: NN is trained to predict binary mask.
  Each pixel of the mask may be between 1 for cancerous region and 0 otherwise.
  Use :doc:`segmentation_targets <../api/unpackers>`.

* **Regression**: NN is trained to predict location, sizes and probability
  of cancer tumor at once. Altogether, there are 7 target values:
  three for location (z, y, x coordinates), three for sizes (z-diam, y-dima, x-diam)
  and one for probability of cancer. Use :doc:`regression_targets <../api/unpackers>`.

* **Classification**: labelling input crops or scans with probability of cancer
  presence. Use :doc:`classification_targets <../api/unpackers>`.

------------------------------------------------------------------------------------

For simplicity, Call batch method `.unpack` with desired argument
(e.g. 'regression_targets') in dataset's `.train_model' method, see example
of training `ResNodule3DNet50` model for classification task:

.. code-block:: python

  from radio import CTImagesMaskedBatch as CT
  from radio.models.tf import ResNodule3DNet50
  from radio import dataset as ds
  from radio.dataset import F

  resnet_config = {
    'num_targets': 1,
    'optimizer': 'Adam',
    'loss': tf.losses.log_loss,
  }

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', ResNodule3DNet50, 'resnet', config=resnet_config)
      .train_model('resnet', feed_dict={'images': F(CT.unpack, component='images')
                                        'labels': F(CT.unpack, component='classification_targets')})
  )

Now train loop can be started:

.. code-block:: python

  (train_dataset >> train_ppl).run(batch_size=16)

In example above `init_model` and `train_model` methods are methods of
ds.Pipeline instances.

**init_model** method is called just once
when pipeline object is being constructed. First argument of this method is
type of model: 'static' or 'dynamic'. Second -- model's class,
third argument -- name of model, last one -- model's configuration dict.
Configuration dictionary may contain parameters that will be used by a model
when it is being built. More information about configuration dictionary, models types
and their interaction with ``ds.Pipeline`` instances
can be found in :doc:`models section <../api/models>`
of dataset package documentation.

**train_model** method accepts name of the model as its first argument and
callable that can be used for unpacking data from batch in a format suitable for
ANN learning. This method is called on every iteration.

Full description ``dataset.Pipeline`` methods that enables interaction with models
can be seen in :mod:`dataset <https://analysiscenter.github.io/dataset/intro/models.html>` package documentation.

The same model can be configured for regression task: the only thing
required is to change number of target values and loss functions
in configuration dictionary. Also, another method for unpacking data from
CTImagesMaskedBatch will be used:

.. code-block:: python

  from radio import CTImagesMaskedBatch as CT
  from radio import dataset as ds
  from radio.models.tf import ResNodule3DNet50, reg_l2_loss

  resnet_config = {
    'num_targets': 7,
    'optimizer': 'Adam',
    'loss': reg_l2_loss
  }

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', ResNodule3DNet50, 'resnet', config=resnet_config)
      .train_model(model_name='resnet', feed_dict={
          'images': F(CT.unpack, component='images'),
          'labels': F(CT.unpack, component='regression_targets')
      })
  )

Same for segmentation:

.. code-block:: python

  from radio import CTImagesMaskedBatch as CT
  from radio import dataset as ds
  from radio.models import Keras3DUNet
  from radio.models.keras.losses import dice_loss, tiversky_loss

  vnet_config = {
    'optimizer': 'Adam',
    'loss': tiversky_loss
  }

  train_ppl = (
    ds.Pipeline()
      .load(fmt='blosc')
      .normalize_hu()
      .init_model('static', Keras3DUNet, 'vnet', config=vnet_config)
      # Keras3DUNet has 'channels_first' dim_ordering
      .train_model(model_name='resnet', feed_dict={
          'images': F(CT.unpack, component='images'),
          'labels': F(CT.unpack, component='segmentation_targets',
                      data_format='channels_first')
      })
  )
