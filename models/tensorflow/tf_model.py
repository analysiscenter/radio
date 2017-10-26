# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-context-manager
""" Contains base class for all tensorflow models. """

import os
import functools
import json
import numpy as np
import tensorflow as tf

from ...dataset.dataset.models.tf import TFModel


class TFModel3D(TFModel):
    """ Base class for all tensorflow models. """

    def get_number_of_trainable_vars(self):
        """ Get number of trainable variable in graph associated with current model. """
        with self.graph.as_default():
            arr = np.asarray([np.prod(self.get_shape(v)) for v in tf.trainable_variables()])
        return np.sum(arr)
