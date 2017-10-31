""" Contains base class for all keras models. """

import os
import functools
import shutil
import pandas as pd
from IPython.display import clear_output
import keras
from keras.models import Model
from ...dataset.dataset.models import BaseModel


class KerasModel(Model, BaseModel):
    """ Base class for all keras models.

    Contains load, dump and compile methods which are shared between all
    keras models;
    Also implements train and predict methods;

    """
    def __init__(self, *args, **kwargs):
        """ Call __init__ of BaseModel not keras.models.Model. """
        BaseModel.__init__(self, *args, **kwargs)

        self._show_metrics = self.get_from_config('show_metrics', False)
        self._test_pipeline = self.get_from_config('test_pipeline', None)
        self._train_metrics_values = []
        self._test_metrics_values = []

    def refresh_metrics(self):
        """ Refresh metrics values. """
        self._train_metrics_values = []
        self._test_metrics_values = []

    @property
    def train_metrics(self):
        """ Return pandas DataFrame containing train metrics. """
        return pd.DataFrame(self._train_metrics_values)

    @property
    def test_metrics(self):
        """ Return pandas DataFrame containing train metrics. """
        return pd.DataFrame(self._test_metrics_values)

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

    def train(self, x=None, y=None, **kwargs):
        """ Wrapper for keras.models.Model.train_on_batch.

        Checks whether feed_dict is None and unpacks it as kwargs
        of keras.models.Model.train_on_batch method.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            x argument of keras.models.Model.train_on_batch method, input of
            neural network.
        y : ndarray(batch_size, ...)
            y argument of keras.models.Model.predict_on_batch method.

        Returns
        -------
        ndarray(batch_size, ...)
            predictions of keras model.

        Raises
        ------
        ValueError if 'x' or 'y'  is None.
        """
        if x is None or y is None:
            raise ValueError("Arguments 'x' and 'y' must not be None")

        prediction = self.train_on_batch(x=x, y=y)
        if not isinstance(prediction, (list, tuple)):
            prediction = (prediction, )

        self._train_metrics_values.append(dict(zip(self.metrics_names, prediction)))
        if self._show_metrics:
            print(self._train_metrics_values[-1])
            clear_output(wait=True)
        return prediction

    def predict(self, x=None, **kwargs):
        """ Wrapper for keras.models.Model.predict_on_batch.

        Checks whether feed_dict is None and unpacks it
        as kwargs of keras.models.Model.predict_on_batch method.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            x argument of keras.models.Model.predict_on_batch method, input of
            neural network.

        Returns
        -------
        ndarray(batch_size, ...)
            predictions of keras model.

        Raises
        ------
        ValueError if 'x' argument is None.
        """
        if x is not None:
            return Model.predict_on_batch(self, x=x)
        else:
            raise ValueError("Argument 'x' must not be None")
        return None

    def test_on_dataset(self, unpacker):
        if self._test_pipeline is None:
            return
        self._test_pipeline.reset_iter()
        metrics_on_test = []
        while True:
            batch = self._test_pipeline.next_batch()
            feed_dict = unpacker(batch)
            y_true = feed_dict.get('y', None)
            y_pred = self.predict(x=feed_dict.get('x', None))
            metrics_on_test.append(self.compute_metrics(y_true, y_pred))
        metrics = pd.DataFrame(metrics_on_test).mean()
        self._test_metrics_values.append(metrics.to_dict(metrics))


    @functools.wraps(Model.load_weights)
    def load(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.load_weights. """
        return Model.load_weights(self, **args, **kwargs)

    @functools.wraps(Model.save_weights)
    def save(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.save_weights. """
        return Model.save_weights(self, *args, **kwargs)
