import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import sys
import warnings
warnings.filterwarnings("ignore")
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('./../lung_cancer')
from radio.preprocessing import CTImagesMaskedBatch as CTIMB
from radio.preprocessing.mip import numba_xip
from radio.dataset import Dataset, Pipeline, FilesIndex, F, V, B, C, Config, Research, Option, KV
from radio.dataset.models.tf import UNet, VNet, GCN
from radio.dataset.models.tf.losses import dice_batch, dice, dice2, dice_batch, dice_batch2


PATH = './blosce/*'
index = FilesIndex(path=PATH, dirs=True, no_ext=False)
dataset = Dataset(index=index, batch_class=CTIMB)
dataset.cv_split(0.9, shuffle=120)
dataset.indices.shape


# MIP maker
def make_component(batch, component, start=0, channels=None, squeeze=False):
    images = batch.unpack(component=component, data_format=None)
    new_images = []
    for i in range(images.shape[0]):
        mode = 0 if squeeze else 0 # 0 - max, 1 - min, 2 - avg, 3 - median
                            # image, depth, mode, step, start
        new_image = numba_xip(images[i], 6, mode, 2, start)

        if channels is None:
            new_image = np.expand_dims(new_image, axis=-1)
        else:
            d = len(new_image) % channels
            if d > 0:
                new_image = new_image[:-d]
            t = np.transpose(new_image, (1, 2, 0))
            t = t.reshape(t.shape[:2] + (-1, channels))
            new_image = np.transpose(t, (2, 0, 1, 3))
            if squeeze:
                new_image = np.amax(new_image, axis=-1, keepdims=True)
        new_images.append(new_image)
    new_images = np.stack(new_images)
    new_images = new_images.reshape((-1,) + new_images.shape[2:])
    return new_images

def make_data(batch, channels=None, start=None):
    if start is None:
        start = np.random.randint(18)
    else:
        start = 0
    batch.nimages = make_component(batch, 'images', start, channels=channels)
    batch.nmasks = make_component(batch, 'masks', start, channels=channels, squeeze=True)


# # Custom train method
def train(batch):
    model = batch.get_model_by_name('net')
    predictions = []
    k = 8  # BATCH SIZE
    l = len(batch.nimages)
    order = np.random.permutation(l)
    for i in range(l // k + (1 if l % k > 0 else 0)):
        loss, preds = model.train(fetches=['loss', 'output_sigmoid'], feed_dict={
            'images': batch.nimages[order[i*k:(i+1)*k]],
            'masks': batch.nmasks[order[i*k:(i+1)*k]],
        })
        predictions.append(preds)
    predictions = np.concatenate(predictions, axis=0)
    batch.pipeline.set_variable('predictions', predictions[order])
    batch.pipeline.set_variable('loss', loss)


# # Custom loss
def logloss(labels, predictions):
    e = 1e-15
    predictions = tf.sigmoid(predictions)
    loss = 10 * labels * tf.log(predictions + e) + (1 - labels) * tf.log(1 - predictions - e)
    loss = -tf.reduce_mean(loss)
    tf.losses.add_loss(loss)
    return loss


# Model saving
def save_model(batch, pipeline):
    model = pipeline.get_model_by_name('net')
    #name = pipeline.config['model_config/loss'].__name__
    name = model.__class__.__name__
    model.save('MIP_e_6_3c/models/' + name)
    print('Model', name, 'saved')


# # Pipelines

root_pipeline = (
    Pipeline()
      .load(fmt='blosc')
      .call(make_data, channels=3)
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
gpus = [dict(model_config=dict(device='/device:GPU:'+str(i))) for i in range(2)]

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
    optimizer=dict(use_locking=True),
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
mr.add_pipeline(root_pipeline << dataset.test, test_pipeline, variables=['loss'], name='test', train_pipeline='train', execute_for=1000)
mr.add_grid_config(op)


# # Run research
mr.run(n_reps=1, n_iters=50000, n_workers=1, n_branches=gpus, name='MIP_e_6_3c')
