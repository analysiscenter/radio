""" Class class for nodules. """

import numpy as np
import pandas as pd

COMPONENTS = {
    'luna': ('seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'),
    'radio': ('id', 'x', 'y', 'z', 'diameter')
}

class Nodules:
    """ Class for storing information about cancer nodules. """
    def __init__(self, nodules=None, names='luna'):
        """
        Parameters
        ----------
        nodules : dict, pd.DataFrame or None
        
        names : str
        """
        if isinstance(nodules, (pd.DataFrame, dict)):
            nodules = {k: np.array(v) for k, v in dict(nodules).items()}
            self._components = self._build_nodules(nodules, names)
        elif type(self) == type(nodules):
            self._components = nodules._components
        elif nodules is None:
            self._components = {key: [] for key in COMPONENTS['radio']}

    def _build_nodules(self, nodules, names):
        components = {key1: nodules[key2] for key1, key2 in zip(COMPONENTS['radio'], COMPONENTS[names])}
        return components

    @property
    def id(self):
        return self._components['id']

    @property
    def x(self):
        return self._components['x']

    @property
    def y(self):
        return self._components['y']

    @property
    def z(self):
        return self._components['z']

    @property
    def diameter_mm(self):
        return self._components['diameter']

    @property
    def center(self):
        return np.stack([self.x, self.y, self.z], axis=1)

    @property
    def size(self):
        return np.stack([self.diameter_mm] * 3, axis=1)

    def set_ids(self, new_ids):
        if len(new_ids) != len(self):
            raise ValueError("'new_ids' must be of length {} but length is {}"
                .format(len(self), len(new_ids)))
        self._components['id'] = np.asarray(new_ids)

    def _ids_to_ints(self, ids):
        self.mapping = {value: key for key, value in enumerate(np.unique(ids))}
        return np.array(list(map(self.mapping.get, ids)))

    def _ints_to_ids(self, ids):
        reverse_mapping = {value: key for key, value in self.mapping.items()}
        return np.array(list(map(reverse_mapping.get, ids)))

    def select(self, _sentinel=None, ids=None, batch=None, indices=None, coords=None, as_df=False):
        """ Select nodules that corresponds to the batch or ids. """
        if _sentinel is not None:
            raise ValueError("Only call `select` with named arguments.")
        if ids is not None:
            _indices = np.in1d(self.id, ids)
        elif batch is not None:
            _indices = np.in1d(self.id, batch.index.indices)
        elif indices is not None:
            return {key: value[indices] for key, value in self._components.items()}
        else:
            raise ValueError("At least one of 'indices' and 'batch' must be not None")

        components = {key: value[_indices] for key, value in self._components.items()}
        if coords is not None:
            components = self._filter_by_coords(components, coords)
        if as_df:
            components = pd.DataFrame(components).set_index('id')
        return components

    def filter(self, _sentinel=None, ids=None, batch=None, indices=None, coords=None):
        """ Filter nodules that corresponds to the batch or indices. """
        if _sentinel is not None:
            raise ValueError("Only call `filter` with named arguments.")
        self._components = self.select(ids=ids, batch=batch, indices=indices, coords=coords)

    def batch_indices(self, batch):
        """ Indices of patients in batch that has nodules. """
        _indices = np.in1d(self.id, batch.index.indices)
        _indices = self.id[_indices]
        return batch.index.get_pos(_indices)

    def extend(self, nodules):
        """ Extend nodules. """
        self._components = {
            key: np.concatenate([getattr(self, key), getattr(nodules, key)], axis=0)
            for key in self._components
        }

    def __len__(self):
        return len(self.id)

    @classmethod
    def assemble(self, *nodules):
        """ Concatenate nodules. """
        new_nodules = Nodules()
        [new_nodules.extend(item) for item in nodules]
        return new_nodules

    def random_sample(self, size):
        """ Select random nodules. """
        indices = np.random.choice(len(self), size, replace=False)
        return self.select(indices=indices)

    def spacing(self, batch):
        """ Get spacing for nodules in batch. """
        return batch.data.spacing[batch.index.get_pos(self.id)]

    def origin(self, batch):
        """ Get origin for nodules in batch. """
        return batch.data.origin[batch.index.get_pos(self.id)]

    def patient_pos(self, batch):
        """ Get patients' indices in batch. """
        return batch.index.get_pos(self.id)

    def center_in_pixels(self, batch):
        """ Get coords of nodules' centers in images. """
        return np.abs(self.center - self.origin(batch)) / self.spacing(batch)
    
    def start_in_pixels(self, batch):
        """ Get nodules' starts in images. """
        start_pix = (self.center_in_pixels(batch) - self.size_in_pixels(batch) / 2)
        return np.rint(start_pix).astype(np.int)

    def size_in_pixels(self, batch):
        """ Get nodules sizes. """
        return np.rint(self.size / self.spacing(batch))
