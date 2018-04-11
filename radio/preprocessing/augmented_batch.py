""" Contains CTImagesAugmentedBatch: masked ct-batch with some augmentation actions """

import numpy as np

from .ct_masked_batch import CTImagesMaskedBatch
from ..dataset import action, Sampler  # pylint: disable=no-name-in-module
from .mask import insert_cropped

class CTImagesAugmentedBatch(CTImagesMaskedBatch):
    """ Masked ct-batch with augmenting actions.

    Adds cutout, additive/multiplicative noise - augmentations.
    """
    @action
    def init_with_ones(self, shape=(32, 64, 64)):
        """ Loader for tests, fills batch with ones.
        """
        self.images = np.ones(shape=(len(self) * shape[0], *shape[1:]))
        self._bounds = np.cumsum((0, ) + (shape[0], ) * len(self))
        return self

    @action
    def cutout(self, positions, sizes, fill_with=0):
        """ Fill a box from each scan with some density-value.

        Parameters:
        -----------
        positions : ndarray
            array of starting positions of boxes, has shape (len(batch), 3).
        size : ndarray
            array of box-sizes, has shape (len(batch), 3).
        fill_with : ndarray, float or string
            value or filling scheme. Value can be float or an array of the shape,
            that can be broadcasted to box-shape. When string, can be either scan-wise
            mean ('mean') or scan-wise minimum/maximum ('min', 'max').
        """
        for i in range(len(self)):
            size, position = sizes[i].astype(np.int64), positions[i].astype(np.int64)
            item = self.get(i, 'images')

            # parse filling scheme
            fill_with = getattr(np, fill_with)(item) if isinstance(fill_with, str) else fill_with
            filled = np.ones(shape=size) * fill_with

            # perform insertion
            insert_cropped(item, filled, position)

        return self

    @action
    def apply_noise(self, noise, op='+'):
        """ For each item apply the noise to the item using op.

        Parameters:
        -----------
        noise : Sampler/ndarray
            1d-sampler/ndarray of shape=(len(batch), item.shape)
        op : str
            operation to perform on item. Can be either '+', '-', '*'
        """
        # prepare noise-array
        all_items = self.images
        noise = noise.sample(size=all_items.size).reshape(all_items.shape) if isinstance(noise, Sampler) else noise

        # parse and apply op in-place
        op_dict = {'+': '__add__', '*': '__mul__', '-': '__sub__'}
        op = op_dict[op]
        all_items[:] = getattr(all_items, op)(noise)

        return self
