import sys
sys.path.append('..')
from dataset import Batch

import os
import numpy as np
#import blosc
import dicom

from preprocessing.auxiliaries import resize_chunk_numba
from preprocessing.auxiliaries import resize_patient_numba
from preprocessing.auxiliaries import get_filter_patient

from preprocessing.mip import image_XIP as XIP
from preprocessing.crop import return_black_border_array as rbba

INPUT_FOLDER = '/notebooks/data/MRT/nci/'
BLOSC_STORAGE = '/notebooks/data/MRT/blosc_preprocessed/'
AIR_HU = -2000


def unpack_blosc(blosc_dir_path):
    """
    unpacker of blosc files
    """
    with open(blosc_dir_path, mode='rb') as f:
        packed = f.read()

    return blosc.unpack_array(packed)


class BatchIterator(object):

    """
    iterator for Batch

    instance of Batch contains concatenated (along 0-axis) patients
        in Batch.data

    consecutive "floors" with numbers from Batch.lower_bounds[i]
        to Batch.upper_bounds[i] belong to patient i


    iterator for Batch iterates over patients
        i-th iteration returns view on i-th patient's data
    """

    def __init__(self, batch):
        self._batch = batch
        self._patient_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._patient_index >= self._batch._lower_bounds.shape[0]:
            raise StopIteration
        else:
            lower = self._batch._lower_bounds[self._patient_index]
            upper = self._batch._upper_bounds[self._patient_index]
            return_value = self._batch._data[lower:upper, :, :]
            self._patient_index += 1
            return return_value


class BatchCt(Batch):

    """
    class for storing batch of Dicom 3d-scans

    Attrs:
        1. data: 3d-array of stacked scans along number_of_slices axis
        2. lower_bounds: 1d-array of first floors for each patient
        3. upper_bounds: 1d, last floors
        4. history: list of preprocessing operations applied to self
        5. patients: list of patients' IDs

    Important methods:
        1. __init__(self, batch_id, patient_names,
                    paths, btype='dicom', **kwargs):
            builds skyscraper of patients

        2. resize(self, new_sizes, order, num_threads):
            transform the shape of all patients to new_sizes
            method is spline iterpolation(order = order)
            the function is multithreaded by num_threads

        3. dump(self, path)
            create a dump of the batch
            in the path-folder

        4. get_filter(self, erosion_radius=7, num_threads=8)
            returns binary-mask for lungs segmentation

        5. segment(self, erosion_radius=2, num_threads=8)
            segments using mask from get_filter()
            that is, sets to hu = -2000 of pixels outside mask
            changes self, returns self

    """

    def load(self, batch_id, all_patients_paths,
             btype='dicom'):
        """
        builds batch of patients

        args:
            batch_id - id of entire batch
            patient_names - patient names/IDs
            paths - paths to files (dicoms/mhd/blosc)
            btype - type of data. 'dicom'|'blosc'|'raw'

        example:

            ###############################################
            ###############################################
            ADD ONE!
            ###############################################
            ###############################################


        ***to do: rewrite initialization with asynchronicity
        """

        #######################################################
        #######################################################
        # somehow make paths from patient_names and input_folder
        #######################################################
        #######################################################

        # auxiliary dictionaries for indexation
        # dictionary index (patient name) -> path for storing his data
        self._patient_name_path = {patient:
            all_patients_paths[patient] for patient in self.index}

        # read, prepare and put 3d-scans in list
        if btype == 'dicom':
            list_of_arrs = self._make_dicom()
        elif btype == 'blosc':
            list_of_arrs = self._make_blosc()
        elif btype == 'raw':
            list_of_arrs = self._make_raw()
        else:
            raise TypeError("Incorrect type of batch source")

        # concatenate scans and initialize patient bounds
        self._initialize_data_and_bounds(list_of_arrs)

    def __init__(self, index):
        """
        common part of initialization from all formats
            -initialization of all attrs
            -creation of empty lists and arrays    
        """

        super().__init__(index)

        self._data = None

        self._upper_bounds = np.array([], dtype=np.int32)
        self._lower_bounds = np.array([], dtype=np.int32)

        self._patient_name_path = dict()

        self._crop_centers = np.array([], dtype=np.int32)
        self._crop_sizes = np.array([], dtype=np.int32)

        self.history = []

    def _make_dicom(self):
        """
        read, prepare and put 3d-scans in list
            given that self contains paths to dicoms in
            self._patient_name_path
        """

        list_of_arrs = []
        for patient in self.index:
            patient_folder = self._patient_name_path[patient]

            list_of_dicoms = [dicom.read_file(os.path.join(patient_folder, s))
                              for s in os.listdir(patient_folder)]

            list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]),
                                reverse=True)
            intercept_pat = list_of_dicoms[0].RescaleIntercept
            slope_pat = list_of_dicoms[0].RescaleSlope

            patient_data = np.stack([s.pixel_array
                                     for s in list_of_dicoms]).astype(np.int16)

            patient_data[patient_data == AIR_HU] = 0

            if slope_pat != 1:
                patient_data = slope_pat * patient_data.astype(np.float64)
                patient_data = patient_data.astype(np.int16)

            patient_data += np.int16(intercept_pat)
            list_of_arrs.append(patient_data)
        return list_of_arrs

    def _make_blosc(self):
        """
        read, prepare and put 3d-scans in list
            given that self contains paths to blosc in
            self._patient_name_path

            *no conversion to hu here
        """
        list_of_arrs = [unpack_blosc(self._patient_name_path[patient])
                        for patient in self.index]

        return list_of_arrs

    def _make_raw(self):
        """
        read, prepare and put 3d-scans in list
            given that self contains paths to raw (see itk library) in
            self._patient_name_path

            *no conversion to hu here
        """
        list_of_arrs = [sitk.GetArrayFromImage(sitk.ReadImage(self._patient_name_path[patient]))
                        for patient in self.index]
        return list_of_arrs

    def _initialize_data_and_bounds(self, list_of_arrs):
        self._data = np.concatenate(list_of_arrs, axis=0)
        list_of_lengths = [len(a) for a in list_of_arrs]
        self._upper_bounds = np.cumsum(np.array(list_of_lengths))
        self._lower_bounds = np.insert(self._upper_bounds, 0, 0)[:-1]
