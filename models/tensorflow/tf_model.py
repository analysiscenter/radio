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
    """ Base class for all tensorflow models. """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        _metrics = self.get_from_config('metrics', [])
        if not isinstance(_metrics, (list, tuple)):
            _metrics = [_metrics]
        self._metrics = _metrics
        self._metrics_values = []
        self._show_metrics = self.get_from_config('show_metrics', False)

    def refresh_metrics(self):
        """ Refresh metrics values. """
        self._metrics_values = []

    @property
    def train_metrics(self):
        """ Return pandas DataFrame containing train metrics. """
        return pd.DataFrame(self._metrics_values)

    def compute_metrics(self, y_true, y_pred):
        """ Compute all attached metrics on train and return result. """
        return {metric.__name__: metric(y_true, y_pred)
                for metric in self._metrics}

    def train(self, x=None, y=None, **kargs):
        """ Train model with data provided. """
        _fetches = ('y_pred', )
        train_output = super().train(_fetches, {'x': x, 'y': y})
        self._metrics_values.append(self.compute_metrics(y, train_output[0]))

        if self._show_metrics:
            print(self.train_metrics.iloc[-1, :])
            clear_output(wait=True)
        return train_output

    def predict(self, x=None, **kargs):
        """ Predict model on data provided. """
        predictions = super().predict(fetches=None, feed_dict={'x': x})
        return predictions

    def test_on_dataset(self, )
