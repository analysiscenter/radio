import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('../../')
from radio import CTImagesMaskedBatch as CTIMB
from radio.dataset import Dataset, Pipeline, FilesIndex, F, V, B, C, Config, L
from radio.dataset.research import Research, Option, KV
from radio.dataset.models.tf import UNet, VNet

# paths to scans in blosc and annotations-table
PATH_SCANS = './blosc/*'
PATH_ANNOTS = './annotations.csv'

# directory for saving models
MODELS_DIR = './trained_models/'

# dataset and annotations-table
index = FilesIndex(path=PATH_SCANS, dirs=True, no_ext=False)
dataset = Dataset(index=index, batch_class=CTIMB)
dataset.split(0.9, shuffle=120)
nodules = pd.read_csv(PATH_ANNOTS)

def train(batch, model='net', minibatch_size=8, mode='max', depth=6, stride=2, start=0, channels=3):
    """ Custom train method. Train a model in minibatches of xips, fetch loss and prediction.
    """
    # training components
    batch.nimages = batch.xip('images', mode, depth, stride, start, channels=channels)
    batch.nmasks = batch.xip('masks', 'max', depth, stride, start, channels=channels, squeeze=True)

    # train model on minibatches, fetch predictions
    model = batch.get_model_by_name(model)
    predictions = []
    bs, mbs = len(batch.nimages), minibatch_size
    order = np.random.permutation(bs)
    num_minibatches = bs // mbs + (1 if bs % mbs > 0 else 0)
    for i in range(num_minibatches):
        loss, preds = model.train(fetches=['loss', 'output_sigmoid'], feed_dict={
            'images': batch.nimages[order[i * mbs:(i + 1) * mbs]],
            'masks': batch.nmasks[order[i * mbs:(i + 1) * mbs]],
        })
        predictions.append(preds)
    predictions = np.concatenate(predictions, axis=0)

    # put predictions into the pipeline
    batch.pipeline.set_variable('predictions', predictions[order])
    batch.pipeline.set_variable('loss', loss)


def logloss(labels, predictions, coeff=10, e=1e-15):
    """ Weighted logloss.
    """
    predictions = tf.sigmoid(predictions)
    predictions = predictions + e       # add separation from zero
    loss = coeff * labels * tf.log(predictions) + (1 - labels) * tf.log(1 - predictions)
    loss = -tf.reduce_mean(loss)
    tf.losses.add_loss(loss)
    return loss


def save_model(batch, pipeline, model='net'):
    """ Function for saving model.
    """
    model = pipeline.get_model_by_name(model)
    name = model.__class__.__name__
    model.save(MODELS_DIR + name)
    print('Model', name, 'saved')

# root, train, test pipelines
root_pipeline = (
    Pipeline()
      .load(fmt='blosc', components=['images', 'spacing', 'origin'])
      .fetch_nodules_info(nodules=nodules)
      .create_mask()
      .run(batch_size=4, shuffle=True, n_epochs=None, prefetch=3, lazy=True)
)

train_pipeline = (
    Pipeline()
      .init_variables(['loss', 'predictions'])
      .init_model('dynamic', C('model'), 'net', C('model_config'))
      .call(train)
)


test_pipeline = (
    Pipeline()
      .init_variables(['loss', 'predictions'])
      .call(save_model, pipeline=C('train_pipeline'))
      .import_model('net', C('train_pipeline'))
      .predict_model('net', fetches='output_sigmoid', feed_dict={
          'images': B('nimages'),
          'masks': B('nmasks'),
      }, save_to=V('predictions'))
)


# define research plan

# model config
model_config = dict(
    inputs=dict(
        images=dict(shape=(256, 256, 3)),
        masks=dict(shape=(256, 256, 1), name='targets')
    ),
    input_block=dict(inputs='images'),
    body=dict(filters=[8, 16, 32, 64]),
    output=dict(ops='sigmoid'),
    loss=logloss,
    optimizer='Adam',
    session=dict(config=tf.ConfigProto(allow_soft_placement=True)),
)


# pipeline config
pipeline_config = Config(model_config=model_config)
train_pipeline.set_config(pipeline_config)


# specify options
p1 = KV('model', 'model')
op = Option(p1, [UNet, VNet])


# define research
mr = Research()
mr.add_pipeline(root_pipeline << dataset.train, train_pipeline, variables=['loss'], name='train')
mr.add_pipeline(root_pipeline << dataset.test, test_pipeline, variables=['loss'], name='test',
                train_pipeline='train', execute_for=1000)
mr.add_grid_config(op)


# run research
mr.run(n_reps=1, n_iters=50000, n_workers=1, n_branches=gpus, name='MIP_research')
