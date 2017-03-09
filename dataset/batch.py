""" Contains basic Batch classes """

import os
import feather


class Batch:
    """ Base Batch class """
    def __init__(self, batch_id, index):
        """ Create batch by subsetting source dataset """
        self.batch_id = batch_id
        self.index = index
        self.data = None

    def load(self, src):
        """ Load data from a file or another data source """
        raise NotImplementedError()

    def dump(self, dst):
        """ Save batch data to disk """
        raise NotImplementedError()


class ArrayBatch(Batch):
    """ Base Batch class for array-like datasets """

    def load(self, src):
        """ Load data from another array """
        self.data = src[self.index]
        return self

    def dump(self, dst):
        """ Save batch data to a file or into another array """
        return self


class DataFrameBatch(Batch):
    """ Base Batch class for datasets stored in pandas DataFrame """

    def load(self, src):
        """ Load batch from a dataframe """
        self.data = src.loc[self.index]
        return self


    def dump(self, dst, fmt='feather', *args, **kwargs):
        """ Save batch data to disk
            dst should point to a directory where all batches will be stored
            in separate files named 'batch_id.format', e.g. '1.csv', '2.csv', etc.
        """
        fname = os.path.join(dst, self.batch_id + '.' + fmt)
        if format == 'feather':
            feather.write_dataframe(self.data, fname, *args, **kwargs)
        elif format == 'csv':
            self.data.to_csv(fname, *args, **kwargs)
        else:
            raise ValueError('Unknown format %s' % fmt)
