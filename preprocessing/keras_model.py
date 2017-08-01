import os
import sys
import shutil
from functools import wraps, partial
import logging
import keras
from keras.models import model_from_json
from .model import BaseModel


logging.basicConfig(format=u'%(levelname)-8s [%(asctime)s] %(message)s',
                    level=logging.INFO)


class KerasModel(BaseModel):

    def __init__(self, name, *args, **kwargs):
        """ Initialize keras model. """
        super().__init__(name, *args, **kwargs)
        self.log = logging.getLogger(name)
        self.name = name
        self.log.info("Building keras model...")
        self.model = self.initialize_model(**kwargs)
        self.log.info("Keras model was build")

    @classmethod
    def initialize_model(cls):
        """ Initialize inner keras model. """
        return None

    @property
    def logger(self):
        """ Get logger for this model. """
        return self.log

    @wraps(keras.models.Model.compile)
    def compile(self, *args, **kwargs):
        """ Compile keras model. """
        self.log.info("Compiling keras model...")
        self.model.compile(*args, **kwargs)
        self.log.info('Model was compiled')

    @wraps(keras.models.Model.train_on_batch)
    def fit(self, *args, **kwargs):
        """ Fit model on input batch and true labels. """
#        self.model.train_on_batch(*args, **kwargs)
        self.model.fit(*args, **kwargs)

    @wraps(keras.models.Model.predict_on_batch)
    def predict(self, *args, **kwargs):
        """ Get predictions on batch x. """
        return self.model.predict_on_batch(*args, **kwargs)

    def load(self, dir_path, *args, **kwargs):
        """ Load model. """
        if not os.path.exists(dir_path):
            raise ValueError("Directory %s does not exists!" % dir_path)

        self.log.info("Loading keras model...")
        with open(os.path.join(dir_path, 'model.json'), 'r') as f:
            model = model_from_json(f.read())
        self.model = model
        self.model.load_weights(os.path.join(dir_path, 'model.h5'))
        self.log.info("Loaded model from %s" % dir_path)

    def load_model(self, path, **kwargs):
        """ Load weights and description of keras model. """
        self.log.info("Loading keras model...")
        self.model = keras.models.load_model(path, custom_objects=kwargs)
        self.log.info("Loaded model from %s" % path)

    def save(self, dir_path, *args, **kwargs):
        """ Save model. """
        with open(os.path.join(dir_path, 'model.json'), 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(os.path.join(dir_path, 'model.h5'))
        self.log.info("Saved model in %s" % dir_path)
