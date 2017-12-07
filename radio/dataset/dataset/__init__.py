""" Dataset enables a fast processing of large dataset using flexible pipelines """
import sys

from .base import Baseset
from .batch import Batch, ArrayBatch, DataFrameBatch
from .batch_image import ImagesBatch, ImagesPILBatch, CROP_00, CROP_CENTER
from .dataset import Dataset
from .pipeline import Pipeline
from .named_expr import B, C, F, V
from .jointdataset import JointDataset, FullDataset
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, model
from .exceptions import SkipBatchException


__version__ = '0.2.1'


if sys.version_info < (3, 5):
    raise ImportError("Dataset module requires Python 3.5 or higher")
