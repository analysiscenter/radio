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
First, you index the scans using the [pipeline](https://analysiscenter.github.io/cardio/intro/pipeline.html), just like that:
```python
	from radio import CTImagesBatch
    from dataset import FilesIndex, Dataset

    dicomix = FilesIndex(path='path/to/dicom/*', no_ext=True) # set up the index
    dicomset = Dataset(index=dicomix, batch_class=CTImagesBatch) # init the dataset of dicom files
```
You may want to resize the scans to equal shape **[128, 256, 256]**,
normalize voxel densities to range **[0, 255]** and dump transformed
scans. Preprocessing like this can be easily done with the following
pipeline, just like that:

```python
    dir_dump = '/path/to/preprocessed/' # preprocessed scans are stored here

    prep_ppl = (dicomset.pipeline() # set up preprocessing workflow
                .load(fmt='dicom')
                .resize(shape=(128, 256, 256))
                .normalize_hu()
                .dump(dir_dump)) # dump results of the preprocessing

    prep_ppl.run(batch_size=20) # run it only now
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

    pipe = get_crops(fmt='raw', shape=(128, 256, 256), nodules_df=nodules, batch_size=20,
                     share=0.6, nodule_shape=(32, 64, 64))

    (pipe >> ctset).gen_batch(batch_size=12, shuffle=True)

    for batch in gen_batches:
        # ...
        # perform net training here
```
See [pipelines section](https://analysiscenter.github.io/lung_cancer/intro/pipelines.html) for more information about
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
    from radio.preprocessing.CTImagesMaskedBatch as CT
    from dataset import F

    training_flow = (ctset.pipeline().
                     .load(fmt='raw')
                     .sample_nodules(nodule_size=(32, 64, 64), batch_size=20) # sample 20 crops from scans
                     .init_model('static',class=DenseNoduleNet, model_name='dnod_net')
                     .train_model(model_name='dnod_net', x=F(CT.unpack, component='images'),
                                  y=F(CT.unpack, component='classification_targets')))

    training_flow.run(batch_size=10)
```
See [models section](https://analysiscenter.github.io/lung_cancer/intro/models.html) for more information about implemented architectures and their application to cancer detection.