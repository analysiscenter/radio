""" Skyscrapper class. """

import numpy as np

class Skyscraper:
    """ Class for saving array of 3d arrays as stacked 3d array. """
    def __init__(self, data=None, shape=None, bounds=None):
        """
        Parameters
        ----------
        data : Skyscraper, list of arrays, array or None
            if Skyscraper it will be copied.
            if None Skyscrapper will be preallocated using shape and bounds

        shape : tuple or None
            shape of input data except one axis

        bounds : array or None
        """
        if isinstance(data, Skyscraper):
            self.data = data.data.copy()
            self.bounds = data.bounds.copy()
        elif data is not None and shape is not None:
            raise ValueError('One and only one of data and shape must be not None')
        elif data is None:
            self.data = np.zeros((bounds[-1], *shape))
            self.bounds = bounds
        elif shape is None:
            if bounds is None:
                self.data = np.concatenate(data, axis=0)
                self.bounds = np.cumsum([0]+[item.shape[0] for item in data])
            else:
                self.data = data
                self.bounds = bounds

    def prealloc(self, shape, bounds):
        """ Preallocate memory. """
        self.data = np.zeros((bounds[-1], *shape))
        self.bounds = bounds

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            return Skyscraper([self[i] for i in np.arange(len(self))[ix]])
        else:
            ix = self._get_index(ix)
            return self.data[self.bounds[ix]: self.bounds[ix+1], ...]

    def __setitem__(self, ix, item):
        ix = self._get_index(ix)
        self.data[self.bounds[ix]: self.bounds[ix+1], ...] = item

    def __len__(self):
        return len(self.bounds) - 1

    def _get_index(self, ix):
        return ix % len(self)

    def append(self, item):
        """ Append element to skyscraper. """
        if self.data is not None:
            self.data = np.concatenate([self.data, item], axis=0)
            self.bounds = np.concatenate([self.bounds, [self.bounds[-1]+item.shape[0]]], axis=0)
        else:
            self.data = item
            self.bounds = np.array([0, item.shape[0]])

    def extend(self, other):
        """ Extend skyscraper by other skyscraper or list of 3d arrays. """
        if self.data is not None:
            if not isinstance(other, Skyscraper):
                other = Skyscraper(other)
            self.data = np.concatenate([self.data, other.data], axis=0)
            self.bounds = np.concatenate([self.bounds, self.bounds[-1]+other.bounds], axis=0)
        else:
            self.data = other.data
            self.bounds = other.bounds

    @classmethod
    def concat(cls, x, y):
        """ Concatenate two skyscrapers. """
        skyscraper = Skyscraper(x)
        skyscraper.extend(y)
        return skyscraper