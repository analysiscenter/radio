""" Module of nn-models for classification/segmentation of
lung-cancer on CT-scans
"""
# from ..dataset import dataset
from .models_batch import CTImagesModels
from .tensorflow.architectures import TFDenseNet
from .tensorflow.architectures import TFResNet
from .tensorflow.architectures import TFDilatedVnet
from .keras.architectures import KerasUnet
from .keras.architectures import KerasResNet50
from .keras.architectures import KerasVGG16
from .tensorflow import TFModel
from .keras import KerasModel
