import os
import sys
from functools import partial
import logging
import argparse
import numpy as np
import pandas as pd

from ..radio.models.tf import DilatedNoduleNet
from ..radio.models.tf.losses import tversky_loss
from ..radio import dataset as ds
from ..radio.dataset import C, V, F, B, Config
from ..radio.dataset.research import Grid, Option, Research

from .utils import get_train_pipeline, get_test_pipeline


def get_unet_research(train_pipeline, test_pipeline)

    base_model_config = dict(
        inputs=dict(
            images={'shape': (32, 64, 64, 1)},
            labels={'shape': (32, 64, 64, 1), 'name': 'targets'}
        ),
        loss=tversky_loss,
        optimizer='Adam'
    )
    config['input_block/inputs'] = 'images'
    config['head/num_classes'] = 1

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


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run research with dilations in unet')
    parser.add_argument('--cancer_crops', type=str, help='Directory with cancerous crops')
    parser.add_argument('--ncancer_crops', type=str, help='Directory with non cancerous crops')
    parser.add_argument('--scans_dir', type=str, help='Directory with test subset scans')
    parser.add_argument('--nodules', type=str, help='Path to annotation file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and prediction')
    parser.add_argument('--shuffle', type=int, default=8, help='Shuffle key for reproducible experiments')
    parser.add_argument('--n_iters', type=int, default=20000, help='Number of iterations to run each pipeline')
    parser.add_argument('--n_reps', type=int, default=5, help='Number of repetions for each pipeline')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of isolated processes')
    args = parser.parse_args()

    # Get batch size from parsed command line
    batch_size = args.batch_size

    # Create indices and datasets with train cancerous and non cancerous crops
    cancerix = ds.FilesIndex(path=os.path.join(args.cancer_crops + '/*'), dirs=True)
    ncancerix = ds.FilesIndex(path=os.path.join(args.ncancer_path + '/*'), dirs=True)
    cancer_set = ds.Dataset(cancerix, batch_class=CTIMB)
    ncancer_set = ds.Dataset(ncancerix, batch_class=CTIMB)

    # Create index and dataset with luna test scans
    luna_test_index = ds.FilesIndex(path=args.scans_dir, no_ext=True)
    luna_test_set = ds.Dataset(luna_test_index, batch_class=CTIMB)

    # Read nodules' annotation dataframe
    nodules = pd.read_csv(args.nodules)

    # Get train pipeline
    train_pipeline = get_train_pipeline(cancer_set, ncancer_set, DilatedNoduleNet, args.shuffle,
                                        batch_sizes=(batch_size // 2, batch_size - batch_size // 2))

    # Get test pipeline that computes metrics on test scans
    test_pipeline = (luna_test_set >> get_test_pipeline(nodules, batch_size)).run(batch_size=2, lazy=True)

    # Create research engine and run it
    research_engine = get_unet_research(train_pipeline, test_pipeline)
    research_engine.run(n_reps=args.n_reps, n_iters=args.n_iters, n_jobs=args.n_jobs, name=args.name)
