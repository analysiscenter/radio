# split_dump on new npcmr-dataset

import os
import sys
sys.path.append('..')
import radio
import radio.annotation
import glob
import numpy as np
import pandas as pd
import pickle as pkl
from radio.dataset import FilesIndex, Dataset, Pipeline, F


# read df containing info about nodules on scans
dataset_info = (
    radio.annotation.read_dataset_info('/notebooks/data/CT/npcmr/*/*/*/*/*/*', index_col=None)
)

# drop duplicates, leave with minimal z-spacing
ds_dropped = (dataset_info
              .sort_values(by=['AccessionNumber', 'SpacingZ']).drop_duplicates(subset='AccessionNumber')
              .set_index('AccessionNumber'))


# set up Index and Dataset for npcmr
ct_index = FilesIndex(ds_dropped.index.values, paths=dict(ds_dropped.loc[:, 'ScanPath']), dirs=True)
ct_dataset = Dataset(ct_index, batch_class=radio.CTImagesMaskedBatch)




# read dumped annots
with open('merged_nodules.pkl', 'rb') as file:
    final = pkl.load(file)

# filter nodules by confidences
filtered = final[final.confidence > 0.02]

# NOTE!!! This is split_dump*, with added zeroing of origin
from radio import split_dump
ct_dataset.cv_split(0.9)

# read histo of nodules locs
with open('histo.pkl', 'rb') as file:
    histo = pkl.load(file)

# split dump pipeline
takeoff = split_dump('/notebooks/data/CT/npcmr_crops/train/cancerous', '/notebooks/data/CT/npcmr_crops/train/noncancerous',
                     filtered, histo, fmt='dicom')
# run the pipeline
print('takeoff...')
(ct_dataset >> takeoff).run()
