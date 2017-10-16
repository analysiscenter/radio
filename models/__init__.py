""" Module of nn-models for classification/segmentation of
lung-cancer on CT-scans
"""
from .models_batch import CTImagesModels, with_model
from .tensorflow.architectures import TFDenseNet
from .tensorflow.architectures import TFResNet
from .tensorflow.architectures import TFDilatedVnet
from .keras.architectures import KerasUnet
from .keras.architectures import KerasResNet50
from .keras.architectures import KerasVGG16
from .keras_model import KerasModel
from .tensorflow import TFModel
