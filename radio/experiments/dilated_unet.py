import os
import sys
from functools import partial
import logging
import argparse
import numpy as np
import pandas as pd

from .. import CTImagesMaskedBatch as CTIMB

from ..models.tf import DilatedNoduleNet
from ..models.tf.losses import tversky_loss
from .. import dataset as ds
from ..dataset import C, V, F, B, Config
from ..dataset.research import Grid, Option, Research

from .utils import get_train_pipeline, get_test_pipeline


def get_unet_research(train_pipeline, test_pipeline):

    base_model_config = dict(
        inputs=dict(
            images={'shape': (32, 64, 64, 1)},
            labels={'shape': (32, 64, 64, 1), 'name': 'targets'}
        ),
        loss=tversky_loss,
        optimizer='Adam'
    )
    base_model_config['input_block/inputs'] = 'images'
    base_model_config['head/num_classes'] = 1

    dilation_rates = Option('body/dilation_rates', values=[
        [None, None, None, None, None],
        [(1, 2), (1, 2), None, None, None],
        [(1, 2), (1, 2), (1, 2), None, None],
        [(1, 2), (1, 2), (1, 2), (1, 2), None],
        [(1, 2, 4), (1, 2, 4), None, None, None],
        [(1, 2, 4), (1, 2, 4), (1, 2, 4), None, None],
        [(1, 2, 4), (1, 2, 4), (1, 2, 4), (1, 2, 4), None]
    ])

    dilation_shares = Option('body/dilation_shares', values=[
        [None, None, None, None, None],
        [(1, 1), (1, 1), None, None, None],
        [(1, 1), (1, 1), (1, 1), None, None],
        [(1, 1), (1, 1), (1, 1), (1, 1), None],
        [(1, 1, 1), (1, 1, 1), None, None, None],
        [(1, 1, 1), (1, 1, 1), (1, 1, 1), None, None],
        [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), None]
    ])

    model_names = Option('name', values=[
        'unet', 'unet_2_depth_2', 'unet_2_depth_3', 'unet_2_depth_4',
        'unet_24_depth_2', 'unet_24_depth_3', 'unet_24_depth_4'
    ])

    options = Option.product(dilation_rates, dilation_shares, model_names)
    config = Config(model_config=base_model_config)
    research_engine = (
        Research()
        .add_pipeline(train_pipeline, variables='loss', config=config, name='train_pipeline')
        .add_pipeline(test_pipeline, variables=['true_stats', 'pred_stats'],
                      config=config, train_pipeline='train_pipeline', name='test_pipeline')
        .add_grid_config(options)
    )
    return research_engine


def run_experiment(cancer_path, ncancer_path, scans_dir, nodules_path,
                   batch_size=4, shuffle=True, n_reps=5, n_iters=20000, n_jobs=1):

    # Create indices and datasets with train cancerous and non cancerous crops
    cancerix = ds.FilesIndex(path=cancer_path, dirs=True)
    ncancerix = ds.FilesIndex(path=ncancer_path, dirs=True)

    cancer_set = ds.Dataset(cancerix, batch_class=CTIMB)
    ncancer_set = ds.Dataset(ncancerix, batch_class=CTIMB)

    luna_test_index = ds.FilesIndex(path=scans_dir, no_ext=True)
    luna_test_set = ds.Dataset(luna_test_index, batch_class=CTIMB)

    # Reading nodules' annotation dataframe
    nodules = pd.read_csv(nodules_path)

    # Get train pipeline
    train_pipeline = get_train_pipeline(cancer_set, ncancer_set, DilatedNoduleNet, shuffle=shuffle,
                                        batch_sizes=(batch_size // 2, batch_size - batch_size // 2))

    # Get test pipeline that computes metrics on test scans
    test_pipeline = (luna_test_set >> get_test_pipeline(nodules, batch_size)).run(batch_size=2, lazy=True)

    # Create research engine and run it
    research_engine = get_unet_research(train_pipeline, test_pipeline)
    research_engine.run(n_reps=n_reps, n_iters=n_iters, n_jobs=n_jobs, name=name)
