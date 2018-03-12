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
from ..models.metrics import dice
from .. import dataset as ds
from ..dataset import C, V, F, B, Config, Pipeline
from ..dataset.research import Grid, Option, Research


LOGGER = logging.getLogger('research')

def get_masks_patches(batch, masks, size=(32, 64, 64), threshold=0.35):
    masks_patches = (
        CTIMB(batch.index)
        .load(fmt='ndarray', masks=masks, images=batch.images,
              origin=batch.origin, spacing=batch.spacing, bounds=batch._bounds)
        .binarize_mask(threshold=threshold)
        .get_patches(size, size, padding='constant', data_attr='masks')
    )
    return masks_patches

def compute_grid_metrics(batch, size=(4, 8, 8),
                         threshold=0.35, scales=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)):
    true_masks, pred_masks = batch.data.masks, batch.predictions
    size = np.array(size)
    dice_dict = {}
    acc_dict = {}
    precision_dict = {}
    recall_dict = {}
    grid = {}
    for scale in scales:
        isize = np.rint(size * scale).astype(np.int)
        true_grid = np.max(get_masks_patches(batch, true_masks, isize, threshold), axis=(1, 2, 3))
        pred_grid = np.max(get_masks_patches(batch, pred_masks, isize, threshold), axis=(1, 2, 3))
        dice_dict[scale] = dice(true_grid, pred_grid)
        #         acc_dict[scale] = accuracy(true_grid, pred_grid)
        #         precision_dict[scale] = precision(true_grid, pred_grid)
        #         recall_dict[scale] = recall(true_grid, pred_grid)
        #         grid[scale] = (true_grid, pred_grid)
    return dice_dict #acc_dict, precision_dict, recall_dict, grid

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
    return result
    #batch.pipeline.get_variable('true_stats').append(result['true_stats'])
    #batch.pipeline.get_variable('pred_stats').append(result['pred_stats'])
    # LOGGER.info('Processed scans: {} / {}'.format((i + 1) * len(batch), len(batch.pipeline.dataset)))


def get_train_pipeline(cancer_set, ncancer_set, model_class, preprocessing, batch_sizes=(2, 2), shuffle=True):
    train_pipeline = combine_crops(cancer_set, ncancer_set, batch_sizes=batch_sizes, shuffle=shuffle)
    if preprocessing is not None:
        train_pipeline += preprocessing
    train_pipeline += (Pipeline()
        .init_variable('loss', init_on_each_run=list)
        .init_variable('pred', init_on_each_run=list)
        .init_model('dynamic', model_class, 'model', config=C('model_config'))
        .train_model('model', fetches='loss', save_to=V('loss'), mode='a',
                     feed_dict={'images': F(lambda batch: batch.unpack('images')),
                                'labels': F(lambda batch: batch.unpack('masks'))})
        .predict_model('model', fetches=['targets', 'predictions'], save_to=V('pred'), mode='a',
                       feed_dict={'images': F(lambda batch: batch.unpack('images')),
                                  'labels': F(lambda batch: batch.unpack('masks'))})
    )
    return train_pipeline

def get_test_pipeline(nodules, test_set, config=None, batch_size=4):
    config = config or dict()
    def partial_compute(batch):
        return compute_test_metrics(batch, nodules=nodules)

    test_pipeline = test_set >> (
        ds.Pipeline()
        #.init_model('dynamic', TFModel, 'model', config={'build': False, **config, 'load': {'path': './without_mixing'}})
        .import_model('model', C('train_pipeline'))
        .init_variable('stats', init_on_each_run=list)
        .load(fmt='raw')
        .unify_spacing(spacing=(1.7, 1.0, 1.0),
                       shape=(400, 512, 512),
                       method='pil-simd',
                       padding='reflect')
        .fetch_nodules_info(nodules)
        .create_mask()
        .normalize_hu()
        .predict_on_scan(model_name='model',
                         strides=(32, 64, 64),
                         crop_shape=(32, 64, 64),
                         batch_size=batch_size,
                         store_to='predictions',
                         show_progress=False)
        #.update_variable('stats', F(partial_compute), mode='a')
        #.call(lambda batch: partial(compute_test_metrics, nodules=nodules))
        .run(lazy=True, batch_size=2)
    )
    return test_pipeline


def get_test_pipeline(nodules, test_set, config=None, batch_size=4):
    def partial_compute(batch):
        return compute_test_metrics(batch, nodules=nodules)

    test_pipeline = test_set >> (
        ds.Pipeline(config=config)
        .import_model('model', C('train_pipeline'))
        #.init_variable('true_stats', init_on_each_run=list)
        #.init_variable('pred_stats', init_on_each_run=list)
        .init_variable('stats', init_on_each_run=list)
        .load(fmt='raw')
        .unify_spacing(spacing=(1.7, 1.0, 1.0),
                       shape=(400, 512, 512),
                       method='pil-simd',
                       padding='reflect')
        .fetch_nodules_info(nodules)
        .create_mask()
        .normalize_hu()
        .predict_on_scan(model_name='model',
                         strides=(32, 64, 64),
                         crop_shape=(32, 64, 64),
                         batch_size=batch_size,
                         store_to='predictions',
                         show_progress=True)
        #.update_variable('stats', F(partial_compute), mode='a')
        #.call(lambda batch: partial(compute_test_metrics, nodules=nodules))
        .init_variable('dice', init_on_each_run=list)
        .update_variable('dice', F(compute_grid_metrics), mode='a')
        .run(lazy=True, batch_size=2)
    )
    return test_pipeline
