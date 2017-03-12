""" Dataset """

from .dsindex import DatasetIndex


class Dataset:
    """ Dataset """
    def __init__(self, index, batch_class=None):
        self.index = self.build_index(index)
        self.batch_class = batch_class

        self.train = None
        self.test = None
        self.validation = None


    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None):
        """ Create Dataset from another dataset with new index
            (usually subset of the source dataset index)
        """
        if index == dataset.index:
            return dataset
        else:
            bcl = batch_class if batch_class is not None else dataset.batch_class
            return cls(index, batch_class=bcl)

    @staticmethod
    def build_index(index):
        """ Create index """
        if isinstance(index, DatasetIndex):
            return index
        else:
            return DatasetIndex(index)


    def cv_split(self, shares=0.8, shuffle=False):
        """ Split the dataset into train, test and validation sub-datasets
        Subsets are available as .train, .test and .validation respectively

        Usage:
           # split into train / test in 80/20 ratio
           ds.cv_split()
           # split into train / test / validation in 60/30/10 ratio
           ds.cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           ds.cv_split([0.5, 0.3, 0.2])
        """
        self.index.cv_split(shares, shuffle)

        self.train = Dataset.from_dataset(self, self.index.train)
        if self.index.test is not None:
            self.test = Dataset.from_dataset(self, self.index.test)
        if self.index.validation is not None:
            self.validation = Dataset.from_dataset(self, self.index.validation)


    def create_batch(self, batch_indices, *args, **kwargs):
        """ Create a batch from given indices """
        return self.batch_class(batch_indices, *args, **kwargs)


    def gen_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Return an object of the batch class """
        for ix_batch in self.index.gen_batch(batch_size, shuffle, one_pass):
            batch = self.create_batch(ix_batch, *args, **kwargs)
            yield batch
