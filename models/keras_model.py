""" Contains base class for all keras models. """

import os
from functools import wraps
import logging
import keras
from keras.models import model_from_json
from .base_model import BaseModel


logging.basicConfig(format=u'%(levelname)-8s [%(asctime)s] %(message)s',
                    level=logging.INFO)


class KerasModel(BaseModel):
    """ Base class for all keras models.

    Contains load, dump and compile methods which are shared between all
    keras models;
    Also implements train_on_batch and predict_on_batch methods;

    """
    def __init__(self, name, *args, **kwargs):
        """ Initialize keras model. """
        super().__init__(name, *args, **kwargs)
        self.log = logging.getLogger(name)
        self.name = name

        self.log.info("Building keras model...")
        self.model = self.initialize_model(**kwargs)  # pylint: disable=assignment-from-none
        self.log.info("Keras model was build")

    @staticmethod
    def initialize_model(*args, **kwargs):
        """ Initialize inner keras model. """
        return None

    @wraps(keras.models.Model.compile)
    def compile(self, *args, **kwargs):
        """ Compile keras model. """
        self.model.compile(*args, **kwargs)

    def train_on_batch(self, x, y_true, **kwargs):
        """ Train model on batch. """
        self.model.train_on_batch(x, y_true)

    def predict_on_batch(self, x, **kwargs):
        """ Get predictions on batch x. """
        return self.model.predict_on_batch(x)

    def load(self, dir_path, *args, **kwargs):
        """ Load model. """
        if not os.path.exists(dir_path):
            raise ValueError("Directory %s does not exists!" % dir_path)

        self.log.info("Loading keras model...")
        with open(os.path.join(dir_path, 'model.json'), 'r') as f:
            model = model_from_json(f.read())
        self.model = model
        self.model.load_weights(os.path.join(dir_path, 'model.h5'))
        self.log.info("Loaded %s model from %s" % (self.name, dir_path))

    def save(self, dir_path, *args, **kwargs):
        """ Save model. """
        with open(os.path.join(dir_path, 'model.json'), 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(os.path.join(dir_path, 'model.h5'))
