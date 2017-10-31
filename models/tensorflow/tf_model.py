# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-context-manager
""" Contains base class for all tensorflow models. """

import os
import functools
import json
import numpy as np
import pandas as pd
from IPython.display import clear_output
import tensorflow as tf

from ...dataset.dataset.models.tf import TFModel


class TFModel3D(TFModel):
    """ Base class for all tensorflow models.

    This class inherits TFModel class from dataset submodule and
    extends it with metrics accumulating methods. Also
    train and predict methods were overloaded:
    train method gets 'x' and 'y',
    while predict gets only 'x' as arguments instead of 'feed_dict'
    and 'fetches' as it was in parent class. It's simplifies interface
    and makes TFModel3D compatible with KerasModel interface.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        _metrics = self.get_from_config('metrics', [])
        if not isinstance(_metrics, (list, tuple)):
            _metrics = [_metrics]
        self._metrics = _metrics
        self._train_metrics_values = []
        self._test_metrics_values = []
        self._test_pipeline = self.get_from_config('test_pipeline', None)
        self._show_metrics = self.get_from_config('show_metrics', False)

    def refresh(self):
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

    def compute_metrics(self, y_true, y_pred):
        """ Compute all attached metrics on train and return result. """
        return {metric.__name__: metric(y_true, y_pred)
                for metric in self._metrics}

    def train(self, x=None, y=None, **kargs):
        """ Train model with data provided.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            numpy array that will be fed into tf.placeholder that can be accessed
            by 'x' attribute of 'self', typically input of neural network.
        y : ndarray(batch_size, ...)
            numpy array that will be fed into tf.placeholder that can be accessed
            by 'y' attribute of 'self'.

        Returns
        -------
        ndarray(batch_size, ...)
            predicted output.
        """
        _fetches = ('y_pred', )
        train_output = super().train(_fetches, {'x': x, 'y': y})
        self._train_metrics_values.append(self.compute_metrics(y, train_output[0]))

        if self._show_metrics:
            print(pd.Series(self._train_metrics_values[-1]))
            clear_output(wait=True)
        return train_output

    def predict(self, x=None, **kargs):
        """ Predict model on data provided.

        Parameters
        ----------
        x : ndarray(batch_size, ....)
            numpy array that will be fed into tf.placeholder that can be accessed
            by 'x' attribute of 'self', typically input of neural network.

        Returns
        -------
        ndarray(batch_size, ...)
            predicted output.
        """
        predictions = super().predict(fetches=None, feed_dict={'x': x})
        return predictions

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
