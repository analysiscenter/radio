""" Contains base class for all keras models. """

import os
import functools
import shutil
import keras
from keras.models import Model
from ..base_model import BaseModel


class KerasModel(Model, BaseModel):
    """ Base class for all keras models.

    Contains load, dump and compile methods which are shared between all
    keras models;
    Also implements train_on_batch and predict_on_batch methods;

    """
    def __init__(self, *args, **kwargs):
        """ Call __init__ of BaseModel not keras.models.Model. """
        BaseModel.__init__(self, *args, **kwargs)

        self._show_metrics = self.get_from_config('show_metrics', False)
        self._train_metrics = []

    @property
    def train_metrics(self):
        """ Return pandas DataFrame containing train metrics. """
        return pd.DataFrame(self._train_metrics)

    def build(self, *args, **kwargs):
        """ Must return inputs and outputs. """
        input_nodes, output_nodes = self._build(**self.config)
        Model.__init__(self, input_nodes, output_nodes)
        self.compile(loss=self.get_from_config('loss', None),
                     optimizer=self.get_from_config('optimizer', 'sgd'),
                     metrics=self.get_from_config('metrics', []))

    def _build(self, *args, **kwargs):
        """ Must return inputs and outputs. """
        raise NotImplementedError("This method must be implemented in ancestor model class")

    def train(self, feed_dict=None, **kwargs):
        """ Wrapper for keras.models.Model.train_on_batch.

        Checks whether feed_dict is None and unpacks it as kwargs
        of keras.models.Model.train_on_batch method.
        """
        if feed_dict is not None:
            prediction = self.train_on_batch(**feed_dict)
            if not isinstance(prediction, (list, tuple)):
                prediction = (prediction, )
            self._train_metrics.append(dict(zip(self.metrics_names, prediction)))
        if self._show_metrics:
            print(self.train_metrics.iloc[-1, :])
            clear_output(wait=True)
        return None

    def predict(self, feed_dict=None, **kwargs):
        """ Wrapper for keras.models.Model.predict_on_batch.

        Checks whether feed_dict is None and unpacks it
        as kwargs of keras.models.Model.predict_on_batch method.
        """
        if feed_dict is not None:
            return Model.predict_on_batch(self, **feed_dict)
        else:
            raise ValueError("Argument feed_dict must not be None")
        return None

    @functools.wraps(Model.load_weights)
    def load(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.load_weights. """
        return Model.load_weights(self, **args, **kwargs)

    @functools.wraps(Model.save_weights)
    def save(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.save_weights. """
        return Model.save_weights(self, *args, **kwargs)

    def load_model(self, path, custom_objects):
        """ Load weights and description of keras model. """
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
