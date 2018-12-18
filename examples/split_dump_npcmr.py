""" Generate crops (using split_dump) from npcmr-dataset. """

import os
import sys

import glob
import numpy as np
import pandas as pd
import pickle as pkl

import radio.annotation
from radio.batchflow import FilesIndex, Dataset, Pipeline, F
from radio import split_dump, CTImagesMaskedBatch

NPCMR_GLOB = '/notebooks/data/CT/npcmr/*/*/*/*/*/*'
MERGED_NODULES_PATH = './merged_nodules.pkl'
HISTO_PATH = './histo.pkl'
NODULE_CONFIDENCE_THRESHOLD = 0.02
TRAIN_SHARE = 0.9
CANCEROUS_CROPS_PATH = '/notebooks/data/CT/npcmr_crops/train/cancerous'
NONCANCEROUS_CROPS_PATH = '/notebooks/data/CT/npcmr_crops/train/noncancerous'

# read df containing info about nodules on scans
dataset_info = (radio.annotation.read_dataset_info(NPCMR_GLOB, index_col='seriesid', filter_by_min_spacing=True,
                                                   load_origin=False))

# set up Index and Dataset for npcmr
ct_index = FilesIndex(dataset_info.index.values, paths=dict(dataset_info.loc[:, 'ScanPath']), dirs=True)
ct_dataset = Dataset(ct_index, batch_class=CTImagesMaskedBatch)

# read dumped annots
with open(MERGED_NODULES_PATH, 'rb') as file:
    merged = pkl.load(file)

# filter nodules by confidences
filtered = merged[merged.confidence > NODULE_CONFIDENCE_THRESHOLD]

ct_dataset.split(TRAIN_SHARE)

# read histo of nodules locs
with open(HISTO_PATH, 'rb') as file:
    histo = pkl.load(file)

# split dump pipeline
sd = split_dump(CANCEROUS_CROPS_PATH, NONCANCEROUS_CROPS_PATH, filtered, histo, fmt='dicom')

# run the pipeline
print('Running split_dump...')
(ct_dataset >> sd).run()
