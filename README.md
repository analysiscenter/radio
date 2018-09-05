# RadIO

**RadIO** is a framework for data science research of computed tomography (CT) imaging.

Main features:
- Asynchronously load **DICOM** and **MetaImage** (mhd/raw) files
- Dump files to [blosc](http://blosc.org/) to compress datasets and thus accelerate loading
- Transform and augment CT-scans in parallel for faster preprocessing
- Create concise chainable workflows with `actions` or use tailored [pipelines](https://analysiscenter.github.io/radio/intro/pipelines.html) for preprocessing or model training
- Train with ease a zoo of state-of-the-art neural networks for classification or semantic segmentation
- Sample crops of any size from CT-scans for comprehensive training
- Customize distribution of crop [locations](https://analysiscenter.github.io/radio/intro/preprocessing.html?highlight=histogram#sample-crops-from-scan) for improved training
- Predict [on the whole scan](https://analysiscenter.github.io/lung_cancer/api/masked_batch.html#radio.preprocessing.ct_masked_batch.CTImagesMaskedBatch.predict_on_scan)

[The documentation](https://analysiscenter.github.io/radio) contains a comprehensive review of RadIO's capabilities. While [tutorials](https://github.com/analysiscenter/radio/tree/master/tutorials) provide ready-to-use code blocks and a practical demonstration of the most important RadIO features.


## Tutorials

There are four tutorials available:

- In the [first](https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.I.ipynb) one you can learn how to set up a dataset of CT-scans and define a basic preprocessing workflow.
- The [second tutorial](https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.II.ipynb) contains in-depth discussion of preprocessing and augmentation actions.
- The [third tutorial](https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.III.ipynb) explains how to generate batches to train a neural network.
- The [fourth tutorial](https://github.com/analysiscenter/radio/tree/master/tutorials/RadIO.IV.ipynb)
will help you configure and train a neural network to detect cancer.


## Preprocess scans with chained actions

Preprocessing-module contains a set of [actions](https://github.com/analysiscenter/dataset) to efficiently prepare a dataset of CT-scans for neural networks training.

Say, you have a bunch of **DICOM** scans with varying shapes.
First, you create an index and define a dataset:
```python
from radio import CTImagesBatch
from dataset import FilesIndex, Dataset

dicom_ix = FilesIndex(path='path/to/dicom/', dirs=True)            # set up the index
dicom_dataset = Dataset(index=dicom_ix, batch_class=CTImagesBatch) # init the dataset of dicom files
```

You may want to resize the scans to equal shape **[128, 256, 256]**,
normalize voxel densities to range **[0, 255]** and dump transformed
scans. This preprocessing can be easily performed with the following
pipeline:

```python
pipeline = (
    dicom_dataset.p
    .load(fmt='dicom')
    .resize(shape=(128, 256, 256))
    .normalize_hu()
    .dump('/path/to/preprocessed/scans/')
)
pipeline.run(batch_size=20)
```

See the [documentation](https://analysiscenter.github.io/radio/intro/preprocessing.html) for the description of
preprocessing actions implemented in the module.


## Preprocess scans using a pre-defined workflow

Pipelines module contains ready-to-use workflows for most frequent tasks.
For instance, if you want to preprocess a dataset of scans named ``dicom_dataset`` and
prepare data for training a neural network, you can simply execute the following
pipeline creator (without spending much time on thinking what actions to choose for
a workflow):

```python
from radio.pipelines import get_crops

nodata_pipeline = get_crops(fmt='raw', shape=(128, 256, 256),
                            nodules=nodules, batch_size=20,
                            share=0.6, nodule_shape=(32, 64, 64))

dicom_pipeline = dicom_dataset >> nodata_pipeline

for batch in dicom_pipeline.gen_batch(batch_size=12, shuffle=True):
    # ...
    # train a model here
```
See [pipelines section](https://analysiscenter.github.io/radio/intro/pipelines.html) for more information about
ready-to-use workflows.


## Adding a neural network to a workflow

`RadIO` contains proven architectures for classification, segmentation and detection, including neural networks designed specifically
for cancer detection (e.g. `DenseNoduleNet` inspired by the state-of-the-art DenseNet, but well suited for 3D CT scans):

```python
from radio.preprocessing import CTImagesMaskedBatch as CTIMB
from radio.models import DenseNoduleNet
from radio.dataset import F

training_pipeline = (
    dicom_dataset.p
      .load(fmt='raw')
      .fetch_nodules_info(nodules_df)
      .create_mask()
      .sample_nodules(nodule_size=(32, 64, 64), batch_size=20)
      .init_model('static', DenseNoduleNet, 'net')
      .train_model('net', feed_dict={
          'images': F(CTIMB.unpack, component='images'),
          'labels': F(CTIMB.unpack, component='classification_targets')
      })
)

training_pipeline.run(batch_size=10, shuffle=True)
```
The [models documentation](https://analysiscenter.github.io/radio/intro/models.html) contains more information about implemented
architectures and their application to cancer detection.


## Installation

> `RadIO` module is in the beta stage. Your suggestions and improvements are very welcome.

> `RadIO` supports python 3.5 or higher.


### Installation as a python package

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/analysiscenter/radio.git#egg=radio

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/analysiscenter/radio.git

After that just import `radio`:
```python
import radio
```


### Installation as a project repository:

When cloning repo from GitHub use flag ``--recursive`` to make sure that ``Dataset`` submodule is also cloned.

    git clone --recursive https://github.com/analysiscenter/radio.git


## Citing RadIO

Please cite RadIO in your publications if it helps your research.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1156363.svg)](https://doi.org/10.5281/zenodo.1156363)

    Khudorozhkov R., Emelyanov K., Koryagin A. RadIO library for data science research of CT images. 2017.

```
@misc{radio_2017_1156363,
  author = {Khudorozhkov R., Emelyanov K., Koryagin A.},
  title  = {RadIO library for data science research of CT images},
  year   = 2017,
  doi    = {10.5281/zenodo.1156363},
  url    = {https://doi.org/10.5281/zenodo.1156363}
}
```
