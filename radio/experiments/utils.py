import os
import sys
from functools import partial
import logging
import argparse
import numpy as np
import pandas as pd


from .. import CTImagesMaskedBatch as CTIMB
from ..pipelines import combine_crops
from ..models.utils import overlap_nodules
from ..models.tf import DilatedNoduleNet
from ..models.tf.losses import tversky_loss
from .. import dataset as ds
from ..dataset import C, V, F, B, Config, Pipeline
from ..dataset.research import Grid, Option, Research


LOGGER = logging.getLogger('research')


def compute_test_metrics(batch, nodules, threshold=0.35):
    batch = batch.fetch_nodules_info(nodules)
    batch_pred  = (
        CTIMB(batch.indices)
        .load(fmt='ndarray', spacing=batch.spacing,
              origin=batch.origin, masks=batch.masks.copy(),
              images=batch.images, bounds=batch._bounds
        )
        .binarize_mask(threshold=threshold)
        .fetch_nodules_from_mask()
    )
    try:
        result = overlap_nodules(batch, batch.nodules, batch_pred.nodules)
    except Exception as e:
        result = {'true_stats': e, 'pred_stats': e}
    batch.pipeline.get_variable('true_stats').append(result['true_stats'])
    batch.pipeline.get_variable('pred_stats').append(result['pred_stats'])
    # LOGGER.info('Processed scans: {} / {}'.format((i + 1) * len(batch), len(batch.pipeline.dataset)))


def get_train_pipeline(cancer_set, ncancer_set, model_class, preprocessing, batch_sizes=(2, 2), shuffle=True):
    train_pipeline = combine_crops(cancer_set, ncancer_set, batch_sizes=batch_sizes, shuffle=shuffle)
    if preprocessing is not None:
        train_pipeline += preprocessing
    train_pipeline += (Pipeline()
        .init_variable('iter', init_on_each_run=0)
        .init_variable('loss', init_on_each_run=list)
        .init_model('dynamic', model_class, 'model', config=C('model_config'))
        .train_model('model', fetches='loss', save_to=V('loss'), mode='a',
                     feed_dict={'images': F(CTIMB.unpack, 'images'),
                                'labels': F(CTIMB.unpack, 'masks')})
        .update_variable('iter', F(lambda x: V('iter').get(x) + 1))
    )
    return train_pipeline


def get_test_pipeline(nodules, test_set, config=None, batch_size=4):
    test_pipeline = test_set >> (
        ds.Pipeline(config=config)
        .import_model('model', C('train_pipeline'))
        .init_variable('true_stats', init_on_each_run=list)
        .init_variable('pred_stats', init_on_each_run=list)
        .load(fmt='raw')
        .unify_spacing(spacing=(1.7, 1.0, 1.0),
                       shape=(400, 512, 512),
                       method='pil-simd',
                       padding='reflect')
        .normalize_hu()
        .resize(shape=(100, 100, 100))
        .predict_on_scan(model_name='model',
                         strides=(32, 64, 64),
                         crop_shape=(32, 64, 64),
                         batch_size=batch_size,
                         show_progress=False)
        .call(lambda batch: partial(compute_test_metrics, nodules=nodules))
        .run(lazy=True, batch_size=2)
    )
    return test_pipeline