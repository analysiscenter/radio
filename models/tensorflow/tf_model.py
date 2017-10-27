# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-context-manager
""" Contains base class for all tensorflow models. """

import os
import functools
import json
import numpy as np
from IPython.display import clear_output
import tensorflow as tf

from ...dataset.dataset.models.tf import TFModel


class TFModel3D(TFModel):
    """ Base class for all tensorflow models. """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._show_metrics = self.get_from_config('show_metrics', False)
        _metrics = self.get_from_config('metrics', [])
        if not isinstance(_metrics, (list, tuple)):
            _metrics = [_metrics]
        self._metrics = _metrics
        self._metrics_values = []

    @property
    def train_metrics(self):
        """ Return pandas DataFrame containing train metrics. """
        return pd.DataFrame(self._metrics_values)

    def compute_metrics(self, y_true, y_pred):
        """ Compute all metrics on train. """
        return {metric.__name__: metric(y_true, y_pred)
                for metric in self._metrics}

    def train(self, fetches=None, feed_dict=None):
        """ Train model with data provided. """
        if fetches is None:
            _fetches = tuple()
        elif not isinstance(fetches, (tuple, list)):
            _fetches = (fetches, )
        _fetches = ('y_pred', *fetches)
        train_output = super().train(_fetches, feed_dict)

        self._metrics_values.append(self.compute_metrics(feed_dict['y'],
                                                         train_output[0]))

        if self._show_metrics:
            print(self._metrics_values.iloc[-1, :])
            clear_output(wait=True)
        return train_output
