""" Contains basic Batch classes """

import os
import threading

import dill
try:
    import blosc
except ImportError:
    pass
import numpy as np
try:
    import pandas as pd
except ImportError:
    pass
try:
    import feather
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    pass

from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, ModelDirectory, any_action_failed
from .dataset import Dataset
from .batch_base import BaseBatch
from .components import MetaComponentsTuple


class Batch(BaseBatch):
    """ The core Batch class """
    _item_class = None
    components = None

    def __init__(self, index, preloaded=None, *args, **kwargs):
        if  self.components is not None and not isinstance(self.components, tuple):
            raise TypeError("components should be a tuple of strings with components names")
        super().__init__(index, *args, **kwargs)
        self._preloaded = preloaded
        self._preloaded_lock = threading.Lock()
        self.pipeline = None

    def deepcopy(self):
        """ Return a deep copy of the batch.

        Constructs a new ``Batch`` instance and then recursively copies all
        the objects found in the original batch, except the ``pipeline``,
        which remains unchanged.

        Returns
        -------
        Batch
        """
        pipeline = self.pipeline
        self.pipeline = None
        dump_batch = dill.dumps(self)
        self.pipeline = pipeline

        restored_batch = dill.loads(dump_batch)
        restored_batch.pipeline = pipeline
        return restored_batch

    @classmethod
    def from_data(cls, index, data):
        """ Create batch from a given dataset """
        # this is roughly equivalent to self.data = data
        if index is None:
            index = np.arange(len(data))
        return cls(index, preloaded=data)

    @classmethod
    def from_batch(cls, batch):
        """ Create batch from another batch """
        return cls(batch.index, preloaded=batch._data)  # pylint: disable=protected-access


    @classmethod
    def merge(cls, batches, batch_size=None):
        """ Merge several batches to form a new batch of a given size

        Parameters
        ----------
        batches : tuple of batches

        batch_size : int or None
            if `None`, just merge all batches into one batch (the rest will be `None`),
            if `int`, then make one batch of `batch_size` and a batch with the rest of data.

        Returns
        -------
        batch, rest : tuple of two batches
        """
        def _make_index(data):
            return DatasetIndex(np.arange(data.shape[0])) if data is not None and data.shape[0] > 0 else None

        def _make_batch(data):
            index = _make_index(data[0])
            return cls(index, preloaded=tuple(data)) if index is not None else None

        if batch_size is None:
            break_point = len(batches)
            last_batch_len = len(batches[-1])
        else:
            break_point = -1
            last_batch_len = 0
            cur_size = 0
            for i, b in enumerate(batches):
                cur_batch_len = len(b)
                if cur_size + cur_batch_len >= batch_size:
                    break_point = i
                    last_batch_len = batch_size - cur_size
                    break
                else:
                    cur_size += cur_batch_len
                    last_batch_len = cur_batch_len

        components = batches[0].components or (None,)
        new_data = list(None for _ in components)
        rest_data = list(None for _ in components)
        for i, comp in enumerate(components):
            if batch_size is None:
                new_comp = [b.get(component=comp) for b in batches[:break_point]]
            else:
                b = batches[break_point]
                last_batch_len_ = b.get_pos(None, comp, b.indices[last_batch_len - 1])
                new_comp = [b.get(component=comp) for b in batches[:break_point]] + \
                           [batches[break_point].get(component=comp)[:last_batch_len_ + 1]]
            new_data[i] = cls.merge_component(comp, new_comp)

            if batch_size is not None:
                rest_comp = [batches[break_point].get(component=comp)[last_batch_len_ + 1:]] + \
                            [b.get(component=comp) for b in batches[break_point + 1:]]
                rest_data[i] = cls.merge_component(comp, rest_comp)

        new_batch = _make_batch(new_data)
        rest_batch = _make_batch(rest_data)

        return new_batch, rest_batch

    @classmethod
    def merge_component(cls, component=None, data=None):
        """ Merge the same component data from several batches """
        _ = component
        if isinstance(data[0], np.ndarray):
            return np.concatenate(data)
        else:
            raise TypeError("Unknown data type", type(data[0]))

    def as_dataset(self, dataset=None):
        """ Makes a new dataset from batch data

        Parameters
        ----------
        dataset: could be a dataset or a Dataset class

        Returns
        -------
        an instance of a class specified by `dataset` arg, preloaded with this batch data
        """
        if dataset is None:
            dataset_class = Dataset
        elif isinstance(dataset, Dataset):
            dataset_class = dataset.__class__
        elif isinstance(dataset, type):
            dataset_class = dataset
        else:
            raise TypeError("dataset should be some Dataset class or an instance of some Dataset class or None")
        return dataset_class(self.index, batch_class=type(self), preloaded=self.data)

    @property
    def indices(self):
        """: numpy array - an array with the indices """
        if isinstance(self.index, DatasetIndex):
            return self.index.indices
        return self.index

    def __len__(self):
        return len(self.index)

    @property
    def data(self):
        """: tuple or named components - batch data """
        if self._data is None and self._preloaded is not None:
            # load data the first time it's requested
            with self._preloaded_lock:
                if self._data is None and self._preloaded is not None:
                    self.load(self._preloaded)
        res = self._data if self.components is None else self._data_named
        return res if res is not None else self._empty_data

    def make_item_class(self, local=False):
        """ Create a class to handle data components """
        # pylint: disable=protected-access
        if self.components is None:
            type(self)._item_class = None
        elif type(self)._item_class is None or not local:
            comp_class = MetaComponentsTuple(type(self).__name__ + 'Components', components=self.components)
            type(self)._item_class = comp_class
        else:
            comp_class = MetaComponentsTuple(type(self).__name__ + 'Components' + str(id(self)),
                                             components=self.components)
            self._item_class = comp_class

    @action
    def add_components(self, components, init=None):
        """ Add new components

        Parameters
        ----------
        components : str or list
            new component names
        init : array-like
            initial component data
        """
        if isinstance(components, str):
            components = (components,)
            init = (init,)
        elif isinstance(components, list):
            components = tuple(components)

        data = self._data
        if self.components is None:
            self.components = components
            data = tuple()
        else:
            self.components = self.components + components
            data = data + tuple(init)
        self.make_item_class(local=True)
        self._data = data

        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_data_named')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            # this warrants that all hidden objects are reconstructed upon unpickling
            setattr(self, k, v)

    @property
    def _empty_data(self):
        return None if self.components is None else self._item_class()   # pylint: disable=not-callable

    def get_pos(self, data, component, index):
        """ Return a position in data for a given index

        Parameters
        ----------
        data : some array or tuple of arrays
            if `None`, should return a position in :attr:`self.data <.Batch.data>`

        components : None, int or str
            - None - data has no components (e.g. just an array or pandas.DataFrame)
            - int - a position of a data component, when components names are not defined
                (e.g. data is a tuple)
            - str - a name of a data component

        index : any
            an index id

        Returns
        -------
        int
            a position in a batch data where an item with a given index is stored

        Notes
        -----
        It is used to read / write data from / to a given component::

            batch_data = data.component[pos]
            data.component[pos] = new_data

        if `self.data` holds a numpy array, then get_pos(None, None, index) should
        just return `self.index.get_pos(index)`

        if `self.data.images` contains BATCH_SIZE images as a numpy array,
        then `get_pos(None, 'images', index)` should return `self.index.get_pos(index)`

        if `self.data.labels` is a dict {index: label}, then `get_pos(None, 'labels', index)` should return index.

        if `data` is not `None`, then you need to know in advance how to get a position for a given index.

        For instance, `data` is a large numpy array, and a batch is a subset of this array and
        `batch.index` holds row numbers from a large arrays.
        Thus, `get_pos(data, None, index)` should just return index.

        A more complicated example of data:

        - batch represent small crops of large images
        - `self.data.source` holds a few large images (e.g just 5 items)
        - `self.data.coords` holds coordinates for crops (e.g. 100 items)
        - `self.data.image_no` holds an array of image numbers for each crop (so it also contains 100 items)

        then `get_pos(None, 'source', index)` should return `self.data.image_no[self.index.get_pos(index)]`.
        Whilst, `get_pos(data, 'source', index)` should return `data.image_no[index]`.
        """
        _ = component
        if data is None:
            pos = self.index.get_pos(index)
        else:
            pos = index
        return pos

    def __getattr__(self, name):
        if self.components is not None and name in self.components:   # pylint: disable=unsupported-membership-test
            attr = getattr(self.data, name)
            return attr
        else:
            raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    def __setattr__(self, name, value):
        if self.components is not None:
            if name == "_data":
                super().__setattr__(name, value)
                if self._item_class is None:
                    self.make_item_class()
                self._data_named = self._item_class(data=self._data)   # pylint: disable=not-callable
            elif name in self.components:    # pylint: disable=unsupported-membership-test
                setattr(self._data_named, name, value)
                super().__setattr__('_data', self._data_named.data)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def put_into_data(self, data, components=None):
        """ Load data into :attr:`_data` property """
        if self.components is None:
            _src = data
        else:
            _src = data if isinstance(data, tuple) else tuple([data])

        _src = self.get_items(self.indices, _src)

        if components is None:
            self._data = _src
        else:
            components = [components] if isinstance(components, str) else components
            for i, comp in enumerate(components):
                setattr(self, comp, _src[i])

    def get_items(self, index, data=None):
        """ Return one or several data items from a data source """
        if data is None:
            _data = self.data
        else:
            _data = data

        if self._item_class is not None and isinstance(_data, self._item_class):
            pos = [self.get_pos(None, comp, index) for comp in self.components]   # pylint: disable=not-an-iterable
            res = self._item_class(data=_data, pos=pos)    # pylint: disable=not-callable
        elif isinstance(_data, tuple):
            comps = self.components if self.components is not None else range(len(_data))
            res = tuple(data_item[self.get_pos(data, comp, index)] if data_item is not None else None
                        for comp, data_item in zip(comps, _data))
        elif isinstance(_data, dict):
            res = dict(zip(_data.keys(), (_data[comp][self.get_pos(data, comp, index)] for comp in _data)))
        else:
            pos = self.get_pos(data, None, index)
            res = _data[pos]
        return res

    def get(self, item=None, component=None):
        """ Return an item from the batch or the component """
        if item is None:
            if component is None:
                res = self.data
            else:
                res = getattr(self, component)
        else:
            if component is None:
                res = self[item]
            else:
                res = self[item]
                res = getattr(res, component)
        return res

    def __getitem__(self, item):
        return self.get_items(item)

    def __iter__(self):
        for item in self.indices:
            yield self[item]

    @property
    def items(self):
        """: list - batch items """
        return [[self[ix]] for ix in self.indices]

    def run_once(self, *args, **kwargs):
        """ Init function for no parallelism
        Useful for async action-methods (will wait till the method finishes)
        """
        _ = self.data, args, kwargs
        return [[]]

    def get_model_by_name(self, model_name):
        """ Return a model specification given its name """
        return ModelDirectory.get_model_by_name(model_name, batch=self)

    def get_errors(self, all_res):
        """ Return a list of errors from a parallel action """
        all_errors = [error for error in all_res if isinstance(error, Exception)]
        return all_errors if len(all_errors) > 0 else None

    @action
    def do_nothing(self, *args, **kwargs):
        """ An empty action (might be convenient in complicated pipelines) """
        _ = args, kwargs
        return self

    @action
    @inbatch_parallel(init='indices')
    def apply_transform(self, ix, dst, src, func, *args, **kwargs):
        """ Apply a function to each item in the batch

        Parameters
        ----------
        dst : str or array
            the destination to put the result in, can be:

            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        src : str or array
            the source to get data from, can be:

            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        func : callable
            a function to apply to each item from the source

        Notes
        -----
        apply_transform does the following (but in parallel)::

            for item in range(len(batch)):
                self.dst[item] = func(self.src[item], *args, **kwargs)
        """
        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least one of dst and src should be an attribute name, not an array")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = self.get(ix, src)
            else:
                pos = self.get_pos(None, dst, ix)
                src_attr = src[pos]
            _args = tuple([src_attr, *args])

        if isinstance(dst, str):
            dst_attr = self.get(component=dst)
            pos = self.get_pos(None, dst, ix)
        else:
            dst_attr = dst
            pos = self.get_pos(None, src, ix)
        dst_attr[pos] = func(*_args, **kwargs)

    @action
    def apply_transform_all(self, dst, src, func, *args, **kwargs):
        """ Apply a function the whole batch at once

        Parameters
        ----------
        dst : str or array
            the destination to put the result in, can be:

            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc
        src : str or array
            the source to get data from, can be:

            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        func : callable
            a function to apply to each item from the source

        Notes
        -----
        apply_transform_all does the following::

            self.dst = func(self.src, *args, **kwargs)
        """

        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least of of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = self.get(component=src)
            else:
                src_attr = src
            _args = tuple([src_attr, *args])

        tr_res = func(*_args, **kwargs)
        if isinstance(dst, str):
            setattr(self, dst, tr_res)
        else:
            dst[:] = tr_res
        return self

    def _get_file_name(self, ix, src, ext):
        if src is None:
            if isinstance(self.index, FilesIndex):
                src = self.index.get_fullpath(ix)
                if self.index.dirs:
                    file_name = os.path.join(src, 'data.' + ext)
                else:
                    file_name = src + '.' + ext
            else:
                raise ValueError("File locations must be specified to dump/load data")
        else:
            file_name = os.path.join(os.path.abspath(src), str(ix) + '.' + ext)
        return file_name

    def _assemble_load(self, all_res, *args, **kwargs):
        _ = all_res, args, kwargs
        return self

    @inbatch_parallel('indices', post='_assemble_load', target='f')
    def _load_blosc(self, ix, src=None, components=None):
        """ Load data from a blosc packed file """
        file_name = self._get_file_name(ix, src, 'blosc')
        with open(file_name, 'rb') as f:
            data = dill.loads(blosc.decompress(f.read()))
            if self.components is None:
                components = (data.keys()[0],)
            else:
                components = tuple(components or self.components)
            item = tuple(data[i] for i in components)
        return item

    @inbatch_parallel('indices', target='f')
    def _dump_blosc(self, ix, dst, components=None):
        """ Save blosc packed data to file """
        file_name = self._get_file_name(ix, dst, 'blosc')
        with open(file_name, 'w+b') as f:
            if self.components is None:
                components = (None,)
                item = (self[ix],)
            else:
                components = tuple(components or self.components)
                item = self[ix].as_tuple(components)
            data = dict(zip(components, item))
            f.write(blosc.compress(dill.dumps(data)))

    def _load_table(self, src, fmt, components=None, *args, **kwargs):
        """ Load a data frame from table formats: csv, hdf5, feather """
        if fmt == 'csv':
            _data = pd.read_csv(src, *args, **kwargs)
        elif fmt == 'feather':
            _data = feather.read_dataframe(src, *args, **kwargs)  # pylint: disable=redefined-variable-type
        elif fmt == 'hdf5':
            _data = pd.read_hdf(src, *args, **kwargs)         # pylint: disable=redefined-variable-type

        # Put into this batch only part of it (defined by index)
        if isinstance(_data, pd.DataFrame):
            _data = _data.loc[self.indices]
        elif isinstance(_data, dd.DataFrame):
            # dask.DataFrame.loc supports advanced indexing only with lists
            _data = _data.loc[list(self.indices)].compute()

        components = tuple(components or self.components)
        for i, comp in enumerate(components):
            setattr(self, comp, _data.iloc[:, i].values)

    @action(use_lock='__dump_table_lock')
    def _dump_table(self, dst, fmt='feather', components=None, *args, **kwargs):
        """ Save batch data to table formats

        Args:
          dst: str - a path to dump into
          fmt: str - format: feather, hdf5, csv
          components: str or tuple - one or several component names
        """
        filename = dst

        components = tuple(components or self.components)
        data_dict = {}
        for comp in components:
            comp_data = self.get(component=comp)
            if isinstance(comp_data, pd.DataFrame):
                data_dict.update(comp_data.to_dict('series'))
            elif isinstance(comp_data, np.ndarray):
                if comp_data.ndim > 1:
                    columns = [comp + str(i) for i in range(comp_data.shape[1])]
                    comp_dict = zip(columns, (comp_data[:, i] for i in range(comp_data.shape[1])))
                    data_dict.update({comp: comp_dict})
                else:
                    data_dict.update({comp: comp_data})
            else:
                data_dict.update({comp: comp_data})
        _data = pd.DataFrame(data_dict)

        if fmt == 'feather':
            feather.write_dataframe(_data, filename, *args, **kwargs)
        elif fmt == 'hdf5':
            _data.to_hdf(filename, *args, **kwargs)   # pylint:disable=no-member
        elif fmt == 'csv':
            _data.to_csv(filename, *args, **kwargs)   # pylint:disable=no-member
        else:
            raise ValueError('Unknown format %s' % fmt)

        return self


    @action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        """ Load data from another array or a file.

        Parameters
        ----------
        src :
            a source (e.g. an array or a file name)

        fmt : str
            a source format, one of None, 'blosc', 'csv', 'hdf5', 'feather'

        components : None or str or tuple of str
            components to load

        *args :
            other parameters are passed to format-specific loaders
        **kwargs :
            other parameters are passed to format-specific loaders
        """
        components = [components] if isinstance(components, str) else components
        if fmt is None:
            self.put_into_data(src, components)
        elif fmt == 'blosc':
            self._load_blosc(src, components=components, **kwargs)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._load_table(src, fmt, components, *args, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def dump(self, dst=None, fmt=None, components=None, *args, **kwargs):
        """ Save data to another array or a file.

        Parameters
        ----------
        dst :
            a destination (e.g. an array or a file name)

        fmt : str
            a destination format, one of None, 'blosc', 'csv', 'hdf5', 'feather'

        components : None or str or tuple of str
            components to load

        *args :
            other parameters are passed to format-specific writers

        *kwargs :
            other parameters are passed to format-specific writers
        """
        components = [components] if isinstance(components, str) else components
        if fmt is None:
            if components is not None and len(components) > 1:
                raise ValueError("Only one component can be dumped into a memory array: components =", components)
            components = components[0] if components is not None else None
            dst[self.indices] = self.get(component=components)
        elif fmt == 'blosc':
            self._dump_blosc(dst, components=components)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._dump_table(dst, fmt, components, *args, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def save(self, *args, **kwargs):
        """ Save batch data to a file (an alias for dump method)"""
        return self.dump(*args, **kwargs)


class ArrayBatch(Batch):
    """ Base Batch class for array-like datasets
    Batch data is a numpy array.
    If components are defined, then each component data is a numpy array
    """
    def _assemble_load(self, all_res, *args, **kwargs):
        _ = args
        if any_action_failed(all_res):
            raise RuntimeError("Cannot assemble the batch", all_res)

        if self.components is None:
            self._data = np.stack([res[0] for res in all_res])
        else:
            components = tuple(kwargs.get('components', None) or self.components)
            for i, comp in enumerate(components):
                _data = np.stack([res[i] for res in all_res])
                setattr(self, comp, _data)
        return self


class DataFrameBatch(Batch):
    """ Base Batch class for datasets stored in pandas DataFrames """
    def _assemble_load(self, all_res, *args, **kwargs):
        """ Build the batch data after loading data from files """
        _ = all_res, args, kwargs
        return self
