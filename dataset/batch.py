""" Contains basic Batch classes """

import os
from binascii import hexlify
import blosc
import numpy as np
import pandas as pd
import feather
from .preprocess import action


class Batch:
    """ Base Batch class """
    def __init__(self, index):
        """ Create batch by subsetting source dataset """
        self.index = index
        self.data = None

    @staticmethod
    def make_filename():
        """ Generate unique filename for the batch """
        random_data = np.random.uniform(0, 1, size=10) * 123456789
        # probability of collision is around 2e-10.
        filename = hexlify(random_data.data)[:8]
        return filename.decode("utf-8")

    @action
    def load(self, src, fmt=None):
        """ Load data from a file or another data source """
        raise NotImplementedError()

    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to disk """
        raise NotImplementedError()


class ArrayBatch(Batch):
    """ Base Batch class for array-like datasets """

    @staticmethod
    def _read_file(path, attr):
        with open(path, 'r' + attr) as file:
            data = file.read()
        return data


    @staticmethod
    def _write_file(path, attr, data):
        with open(path, 'w' + attr) as file:
            file.write(data)


    @action
    def load(self, src, fmt=None):
        """ Load data from another array or a file """

        # Read the whole source
        if fmt is None:
            _data = src
        elif fmt == 'blosc':
            packed_array = self._read_file(src, 'b')
            _data = blosc.unpack_array(packed_array)
        else:
            raise ValueError("Unknown format " + fmt)

        # But put into this batch only part of it (defined by index)
        try:
            # this creates a copy of the source data (perhaps view could be more efficient)
            self.data = _data[self.index]
        except TypeError:
            raise TypeError('Source is expected to be array-like')

        return self


    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to a file or into another array """
        filename = self.make_filename()
        fullname = os.path.join(dst, filename + '.' + fmt)

        if fmt is None:
            dst = self.data
        elif fmt == 'blosc':
            packed_array = blosc.pack_array(self.data)
            self._write_file(fullname, 'b', packed_array)
        else:
            raise ValueError("Unknown format " + fmt)
        return self


class DataFrameBatch(Batch):
    """ Base Batch class for datasets stored in pandas DataFrames """

    @action
    def load(self, src, fmt=None, *args, **kwargs):
        """ Load batch from a dataframe """
        # pylint: disable=no-member
        # Read the whole source
        if fmt is None:
            dfr = src
        elif fmt == 'feather':
            dfr = feather.read_dataframe(src, *args, **kwargs)
        elif fmt == 'hdf5':
            dfr = pd.read_hdf(src, *args, **kwargs) # pylint: disable=redefined-variable-type
        elif fmt == 'csv':
            dfr = pd.read_csv(src, *args, **kwargs)
        else:
            raise ValueError('Unknown format %s' % fmt)

        # But put into this batch only part of it (defined by index)
        self.data = dfr.loc[self.index]

        return self


    @action
    def dump(self, dst, fmt='feather', *args, **kwargs):
        """ Save batch data to disk
            dst should point to a directory where all batches will be stored
            as separate files named 'batch_id.format', e.g. '1.csv', '2.csv', etc.
        """
        filename = self.make_filename()
        fullname = os.path.join(dst, filename + '.' + fmt)

        if fmt == 'feather':
            feather.write_dataframe(self.data, fullname, *args, **kwargs)
        elif fmt == 'hdf5':
            self.data.to_hdf(fullname, *args, **kwargs)
        elif fmt == 'csv':
            self.data.to_csv(fullname, *args, **kwargs)
        else:
            raise ValueError('Unknown format %s' % fmt)
        return self
