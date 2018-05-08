import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import sys
import warnings
warnings.filterwarnings("ignore")
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('../../')
from radio import CTImagesMaskedBatch as CTIMB
from radio.dataset import Dataset, Pipeline, FilesIndex, F, V, B, C, Config, L
from radio.dataset.research import Research, Option, KV
from radio.dataset.models.tf import UNet, VNet, GCN
from radio.dataset.models.tf.losses import dice_batch, dice, dice2, dice_batch, dice_batch2

# npcmr in blosc
PATH_NPCMR_SCANS = '/notebooks/ct/npcmr_blosc/*'
PATH_NPCMR_ANNOTS = '/notebooks/ct/annotations/merged_nodules.pkl'
index = FilesIndex(path=PATH_NPCMR_SCANS, dirs=True, no_ext=False)
dataset = Dataset(index=index, batch_class=CTIMB)
dataset.split(0.9, shuffle=120)
dataset.indices.shape

# custom train method
def train(batch, model='net', minibatch_size=8, mode='max', depth=6, stride=2, start=0, channels=3):
    # training components
    batch.nimages = batch.xip_component('images', mode, depth, stride, start, channels)
    batch.nmasks = batch.xip_component('masks', 'max', depth, stride, start, channels, squeeze=True)

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


# custom loss: rebalanced logloss with sigmoids
def logloss(labels, predictions, coeff=10, e=1e-15):
    predictions = tf.sigmoid(predictions)
    predictions = predictions + e       # add separation from zero
    loss = coeff * labels * tf.log(predictions) + (1 - labels) * tf.log(1 - predictions)
    loss = -tf.reduce_mean(loss)
    tf.losses.add_loss(loss)
    return loss


# function for saving model
def save_model(batch, pipeline, model='net'):
    model = pipeline.get_model_by_name(model)
    #name = pipeline.config['model_config/loss'].__name__
    name = model.__class__.__name__
    model.save('MIP_e_6_3c/models/' + name)
    print('Model', name, 'saved')

# read df with confidenced nodules info
with open(PATH_NPCMR_ANNOTS, 'rb') as file:
    confidenced = pkl.load(file)

# nodules filtering function
def filter_nodules(confidenced, threshold=0.02):
    return confidenced[confidenced.confidence > threshold]

# root, train, test pipelines
root_pipeline = (
    Pipeline()
      .load(fmt='blosc', components=['images', 'spacing', 'origin'])
      .init_variable('filtered')
      .update_variable('filtered', L(filter_nodules, confidenced=confidenced))
      .fetch_nodules_info(nodules=V('filtered'))
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


# # List available GPUs
gpus = [dict(model_config=dict(device='/device:GPU:' + str(i))) for i in range(2)]

# # Define research plan

# ## Model config
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


# ## Pipeline config
pipeline_config = Config(model_config=model_config)
train_pipeline.set_config(pipeline_config)


# ## Specify options
#p1 = KV('model_config/loss', 'loss')
#op = Option(p1, [logloss, dice, dice2])

p1 = KV('model', 'model')
op = Option(p1, [UNet, VNet]) #, GCN])


# ## Define research

mr = Research()
mr.add_pipeline(root_pipeline << dataset.train, train_pipeline, variables=['loss'], name='train')
mr.add_pipeline(root_pipeline << dataset.test, test_pipeline, variables=['loss'], name='test',
                train_pipeline='train', execute_for=1000)
mr.add_grid_config(op)


# # Run research
mr.run(n_reps=1, n_iters=50000, n_workers=1, n_branches=gpus, name='MIP_e_6_3c')
