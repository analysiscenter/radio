""" FullDataset """
from .dataset import Dataset
from .preprocess import Preprocessing


class FullDataset:
    """ Dataset which includes data dataset and target dataset """
    def __init__(self, data, target):
        self.data = data
        if isinstance(data, Dataset):
            self.index = data.index
        elif isinstance(data, Preprocessing):
            self.index = data.dataset.index
        else:
            raise TypeError("Data should be Dataset or Preprocessing")

        if not isinstance(target, Dataset) and not isinstance(target, Preprocessing):
            raise TypeError("Target should be Dataset or Preprocessing")
        self.target = target

        # TODO: check if data and target indices are compatible
        self.batch_generator = None


    def gen_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Generate pairs of batches from data and target """
        for ix_batch in self.index.gen_batch(batch_size, shuffle, one_pass):
            data_batch = self.data.create_batch(ix_batch, *args, **kwargs)
            target_batch = self.target.create_batch(ix_batch, *args, **kwargs)
            yield data_batch, target_batch


    def next_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Return a pair of batches from data and target """
        if self.batch_generator is None:
            self.batch_generator = self.gen_batch(batch_size, shuffle=shuffle, one_pass=one_pass, *args, **kwargs)
        batch = next(self.batch_generator)
        return batch
