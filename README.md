# RadIO

`RadIO` is a framework for batch-processing of computational
tomography (CT)-scans for deep learning experiments.

The framework allows you to:
- preprocess scans in a blink of an eye: load data asynchronously from disk, resize them in parallel
- set up preprocessing workflows in a few lines of code
- train with ease a zoo of high-performing neural nets for cancer detection

## Preprocess scans using implemented actions
Preprocessing-module contains a set of [actions](https://github.com/analysiscenter/dataset), that allow to efficiently prepare a dataset of CT-scans for training neural nets.
Say, you have a bunch of scans in **DICOM** format with varying shapes.
First, you index the scans using the [pipeline](https://analysiscenter.github.io/radio/intro/pipeline.html), just like that:
```python
from radio import CTImagesBatch as CTIB
from dataset import FilesIndex, Dataset

dicomix = FilesIndex(path='path/to/dicom/*', no_ext=True) # set up the index
dicomset = Dataset(index=dicomix, batch_class=CTIB) # init the dataset of dicom files
```
You may want to resize the scans to equal shape **[128, 256, 256]**,
normalize voxel densities to range **[0, 255]** and dump transformed
scans. Preprocessing like this can be easily done with the following
pipeline, just like that:

```python
prep_ppl = (
    dicomset
    .pipeline()
    .load(fmt='dicom')
    .resize(shape=(128, 256, 256))
    .normalize_hu()
    .dump('/path/to/preprocessed/scans/')
)
prep_ppl.run(batch_size=20)
```

See the [documentation](https://analysiscenter.github.io/radio/intro/preprocessing.html) for the description of
preprocessing actions implemented in the module.

## Preprocess scans using a workflow from the box
Pipelines-module contains ready-to-use workflows for most frequent tasks.
E.g. if you want to preprocess dataset of scans named ``ctset`` and
prepare data for training a net, you can simply execute the following
pipeline-creator (without spending time on thinking how to chain actions in
a workflow):

```python
from radio.pipelines import get_crops

pipe = get_crops(fmt='raw', shape=(128, 256, 256),
                 nodules_df=nodules, batch_size=20,
                 share=0.6, nodule_shape=(32, 64, 64))

(ctset >> pipe).gen_batch(batch_size=12, shuffle=True)

for batch in gen_batches:
    # ...
    # perform net training here
```
See [pipelines section](https://analysiscenter.github.io/radio/intro/pipelines.html) for more information about
ready-made workflows.

## Adding a neural-net model to a workflow
Contains neural nets' architectures for task of classification,
segmentation and detection. E.g., ``DenseNoduleNet``, an architecutre,
inspired by DenseNet, but suited for 3D scans.:
```python
from radio.models import DenseNoduleNet
```

Using the architectures from Models, one can train deep learning systems
for cancer detection. E.g., initialization and training of a new DenseNoduleNet
on scan crops of shape **[32, 64, 64]** can be implemented as follows:
```python
from radio.preprocessing import CTImagesMaskedBatch as CTIMB
from dataset import F

training_flow = (
    ctset
    .pipeline()
    .load(fmt='raw')
    .sample_nodules(nodule_size=(32, 64, 64), batch_size=20)
    .init_model(mode='static', model_class=DenseNoduleNet,
                model_name='dnod_net')
    .train_model('dnod_net', feed_dict={
        'images': F(CTIMB.unpack, component='images'),
        'labels': F(CTIMB.unpack, component='classification_targets')
    })
)

training_flow.run(batch_size=10)

```
See [models section](https://analysiscenter.github.io/radio/intro/models.html) for more information about implemented architectures and their application to cancer detection.

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

    Khudorozhkov R., Emelyanov K., Koryagin A., Ushakov A. RadIO library for data science research of CT images. 2017.