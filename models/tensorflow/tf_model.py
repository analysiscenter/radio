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
