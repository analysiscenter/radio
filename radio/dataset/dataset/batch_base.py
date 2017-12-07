""" Contains the base batch class """
from binascii import hexlify
import numpy as np


class BaseBatch:
    """ Base class for batches
    Required to solve circular module dependencies
    """
    def __init__(self, index, *args, **kwargs):
        _ = args, kwargs
        self.index = index
        self._data_named = None
        self._data = None
        self.pipeline = None

    @staticmethod
    def make_filename():
        """ Generate unique filename for the batch """
        random_data = np.random.uniform(0, 1, size=10) * 123456789
        # probability of collision is around 2e-10.
        filename = hexlify(random_data.data)[:8]
        return filename.decode("utf-8")
