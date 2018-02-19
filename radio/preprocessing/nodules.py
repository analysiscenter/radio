import numpy as np
from numba import jitclass
from numba import int32, float32, float64
import pandas as pd

items = [
    ('id', int32[:]),
    ('x', float64[:]),
    ('y', float64[:]),
    ('z', float64[:]),
    ('diameter', float64[:]),
]

class Nodules:
    def __init__(self, nodules):
        nodules = {k: np.array(v) for k, v in dict(nodules).items()}
        nodules['seriesuid'] = self._ids_to_ints(nodules['seriesuid'])
        self._nodules_numba = NodulesNumba(**nodules)

    @property
    def x(self):
        return self.nodules.x

    @property
    def y(self):
        return self.nodules.y

    @property
    def z(self):
        return self.nodules.z

    @property
    def id(self):
        return self._ints_to_ids(self.nodules.id)

    @property
    def diameter(self):
        return self.nodules.diameter

    def _ids_to_ints(self, ids):
        self.mapping = {value: key for key, value in enumerate(np.unique(ids))}
        return np.array(list(map(self.mapping.get, ids)))

    def _ints_to_ids(self, ids):
        reverse_mapping = {value: key for key, value in self.mapping.items()}
        return np.array(list(map(reverse_mapping.get, ids)))

    def get_nodules(self):



@jitclass(items)
class NodulesNumba:
    def __init__(self, seriesuid, coordX, coordY, coordZ, diameter_mm):
        self._build_nodules(seriesuid, coordX, coordY, coordZ, diameter_mm)

    def _build_nodules(self, patient_id, x, y, z, diameter):
        self.id = patient_id
        self.x = x
        self.y = y
        self.z = z
        self.diameter = diameter

    def get_nodules(self, patient_id):
        indices = [i for i in len(self.id) if patient_id == self.id[i]]
        nodules = NodulesNumba(
            self.id[indices],
            self.x[indices],
            self.y[indices],
            self.z[indices],
            self.diameter[indices]
        )
        return nodules