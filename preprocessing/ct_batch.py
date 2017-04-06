""" contains Batch class for storing Ct-scans """

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.append('..')

import numpy as np
import blosc
import dicom
import SimpleITK as sitk

from dataset import Batch, action


from .resize import resize_patient_numba
from .segment import get_mask_patient

from .mip import image_xip as xip
from .crop import return_black_border_array as rbba

AIR_HU = -2000
DARK_HU = -2000


def read_unpack_blosc(blosc_dir_path):
    """
    read, unpack blosc file
    """
    with open(blosc_dir_path, mode='rb') as file:
        packed = file.read()

    return blosc.unpack_array(packed)


class CTImagesBatch(Batch):

    """
    class for storing batch of CT(computed tomography) 3d-scans.
        Derived from base class Batch


    Attrs:
        1. index: array of PatientIDs. Usually, PatientIDs are strings
        2. _data: 3d-array of stacked scans along number_of_slices axis
        3. _lower_bounds: 1d-array of first floors for each patient
        4. _upper_bounds: 1d, last floors
        5. history: list of preprocessing operations applied to self
            when method with @action-decorator is called
            info about call should be appended to history

        6. _patient_index_path: index (usually, patientID) -> storage for this patient
                storage is either directory (dicom-case) or file: .blk for blosc
                or .raw for mhd
        7. _patient_index_number: index -> patient's order in batch
                of use for iterators and indexation


    Important methods:
        1. __init__(self, index):
            basic initialization of patient
            in accordance with Batch.__init__
            given base class Batch

        2. load(self, all_patients_paths,
                btype='dicom'):
            builds skyscraper of patients given
            correspondance patient index -> storage
            and type of data to build from
            returns self

        2. resize(self, new_sizes, order, num_threads):
            transform the shape of all patients to new_sizes
            method is spline iterpolation(order = order)
            the function is multithreaded in num_threads
            returns self

        3. dump(self, path)
            create a dump of the batch
            in the path-folder
            returns self

        4. get_mask(self, erosion_radius=7, num_threads=8)
            returns binary-mask for lungs segmentation
            the larger erosion_radius
            the lesser the resulting lungs will be
            * returns mask, not self

        5. segment(self, erosion_radius=2, num_threads=8)
            segments using mask from get_mask()
            that is, sets to hu = -2000 of pixels outside mask
            changes self, returns self

    """

    def __init__(self, index):
        """
        common part of initialization from all formats:
            -execution of Batch construction
            -initialization of all attrs
            -creation of empty lists and arrays

        attrs:
            index - ndarray of indices
            dtype is likely to be string
        """

        super().__init__(index)

        self._data = None

        self._upper_bounds = np.array([], dtype=np.int32)
        self._lower_bounds = np.array([], dtype=np.int32)

        self._patient_index_path = dict()
        self._patient_index_number = dict()

        self._crop_centers = np.array([], dtype=np.int32)
        self._crop_sizes = np.array([], dtype=np.int32)

        self.history = []

    @action
    def load(self, src=None, fmt='dicom', upper_bounds=None): # pylint: disable=arguments-differ
        """
        builds batch of patients

        args:
            src - path to files (dicoms/mhd/blosc), if None then read files from the location defined in the index
            fmt - type of data.
                Can be 'dicom'|'blosc'|'raw'|'ndarray'

        Dicom example:

            # initialize batch for storing batch of 3 patients
            # with following IDs
            index = FilesIndex(path="/some/path/*.dcm", no_ext=True)
            batch = CTImagesBatch(index)
            batch.load("/data/to/files", fmt='dicom')

        Ndarray example:
            # source_array stores a batch (concatted 3d-scans, skyscraper)
            # say, ndarray with shape (400, 256, 256)

            # source_ubounds stores ndarray of last floors for each patient
            # say, source_ubounds = np.asarray([100, 400])
            batch.load(src=source_array, fmt='ndarray', upper_bounds=source_ubounds)

        ***to do: rewrite initialization with asynchronicity
        """

        # if ndarray. Might be better to put this into separate function
        if fmt == 'ndarray':
            lower_bounds = np.insert(upper_bounds, 0, 0)[:-1]
            list_of_arrs = [src[lower_bounds[i]:upper_bounds[i], :, :] for i in range(len(upper_bounds))]
        elif fmt == 'dicom':
            list_of_arrs = self._load_dicom()
        elif fmt == 'blosc':
            list_of_arrs = self._load_blosc()
        elif fmt == 'raw':
            list_of_arrs = self._load_raw()
        else:
            raise TypeError("Incorrect type of batch source")

        # concatenate scans and initialize patient bounds
        self._initialize_data_and_bounds(list_of_arrs)

        # add info in self.history
        info = {}
        info['method'] = 'load'
        info['params'] = {}
        self.history.append(info)

        return self

    def _load_dicom(self):
        """
        read, prepare and put 3d-scans in a list

        Important operations performed here:
         - conversion to hu using meta from dicom-scans
        """

        list_of_arrs = []
        for patient in self.indices:
            patient_folder = self.index.get_fullpath(patient)

            list_of_dicoms = [dicom.read_file(os.path.join(patient_folder, s))
                              for s in os.listdir(patient_folder)]

            list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]),
                                reverse=True)
            intercept_pat = list_of_dicoms[0].RescaleIntercept
            slope_pat = list_of_dicoms[0].RescaleSlope

            patient_data = np.stack([s.pixel_array for s in list_of_dicoms]).astype(np.int16)

            patient_data[patient_data == AIR_HU] = 0

            if slope_pat != 1:
                patient_data = slope_pat * patient_data.astype(np.float64)
                patient_data = patient_data.astype(np.int16)

            patient_data += np.int16(intercept_pat)
            list_of_arrs.append(patient_data)
        return list_of_arrs

    def _load_blosc(self):
        """
        read, prepare and put 3d-scans in list

            *no conversion to hu here
        """
        list_of_arrs = [read_unpack_blosc(
        	os.path.join(self.index.get_fullpath(patient)), 'data.blk') for patient in self.indices]

        return list_of_arrs

    def _load_raw(self):
        """
        read, prepare and put 3d-scans in list

            *no conversion to hu here
        """
        list_of_arrs = [sitk.GetArrayFromImage(sitk.ReadImage(self.index.get_fullpath(patient)))
                        for patient in self.indices]
        return list_of_arrs

    def _initialize_data_and_bounds(self, list_of_arrs):
        """
        put the list of 3d-scans into self._data
        fill in self._upper_bounds and
            self._lower_bounds accordingly

        args:
            self
            list_of_arrs: list of 3d-scans
        """
        # make 3d-skyscraper from list of 3d-scans
        self._data = np.concatenate(list_of_arrs, axis=0)

        # set floors for each patient
        list_of_lengths = [len(a) for a in list_of_arrs]
        self._upper_bounds = np.cumsum(np.array(list_of_lengths))
        self._lower_bounds = np.insert(self._upper_bounds, 0, 0)[:-1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """
        indexation of patients by []

        args:
            self
            index - can be either number (int) of patient
                         in self from [0,..,len(self.index) - 1]
                    or index from self.index
        """
        if isinstance(index, int):
            if index < self._lower_bounds.shape[0] and index >= 0:
                lower = self._lower_bounds[index]
                upper = self._upper_bounds[index]
                return self._data[lower:upper, :, :]
            else:
                raise IndexError(
                    "Index of patient in the batch is out of range")

        else:
        	ind_pos = self.index.get_pos(index)
            lower = self._lower_bounds[ind_pos]
            upper = self._upper_bounds[ind_pos]
            return self._data[lower:upper, :, :]

    @property
    def crop_centers(self):
        """
        returns centers of crop for all scans
        """
        if not self._crop_centers:
            self._crop_params_patients()
        return self._crop_centers

    @property
    def crop_sizes(self):
        """
        returns window sizes for crops
        """
        if not self._crop_sizes:
            self._crop_params_patients()
        return self._crop_sizes

    def _crop_params_patients(self, num_threads=8):
        """
        calculate params for crop
        """
        with ThreadPoolExecutor(max_workers=num_threads) as executor:

            threads = [executor.submit(rbba, pat) for pat in self]
            crop_array = np.array([t.result() for t in threads])

            self._crop_centers = crop_array[:, :, 2]
            self._crop_sizes = crop_array[:, :, : 2]

    @action
    def make_xip(self, step: int=2, depth: int=10,
                 func: str='max', projection: str='axial',
                 num_threads: int=4, verbose: bool=False) -> "Batch":
        """
        This function takes 3d picture represented by np.ndarray image,
        start position for 0-axis index, stop position for 0-axis index,
        step parameter which represents the step across 0-axis and, finally,
        depth parameter which is associated with the depth of slices across
        0-axis made on each step for computing MEAN, MAX, MIN
        depending on func argument.
        Possible values for func are 'max', 'min' and 'avg'.
        Notice that 0-axis in this annotation is defined in accordance with
        projection argument which may take the following values: 'axial',
        'coroanal', 'sagital'.
        Suppose that input 3d-picture has axis associations [z, x, y], then
        axial projection doesn't change the order of axis and 0-axis will
        be correspond to 0-axis of the input array.
        However in case of 'coronal' and 'sagital' projections the source tensor
        axises will be transposed as [x, z, y] and [y, z, x]
        for 'coronal' and 'sagital' projections correspondingly.
        """
        args_list = []
        for lower, upper in zip(self._lower_bounds, self._upper_bounds):

            args_list.append(dict(image=self._data,
                                  start=lower,
                                  stop=upper,
                                  step=step,
                                  depth=depth,
                                  func=func,
                                  projection=projection,
                                  verbose=verbose))

        upper_bounds = None
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list_of_lengths = []

            mip_patients = list(executor.map(lambda x: xip(**x), args_list))
            for patient_mip_array in mip_patients:
                axis_null_size = patient_mip_array.shape[0]
                list_of_lengths.append(axis_null_size)

            upper_bounds = np.cumsum(np.array(list_of_lengths))

        # construct resulting batch with MIPs
        batch = type(self)(self.index)
        batch.load(fmp='ndarray', src=np.concatenate(mip_patients, axis=0),
                   upper_bounds=upper_bounds)

        return batch

    @action
    def resize(self, num_slices_new=128, num_x_new=256,
               num_y_new=256, order=3, num_threads=8):
        """
        performs resize (change of shape) of each CT-scan in the batch.
            When called from Batch, changes Batch
            returns self

            params: (num_slices_new, num_x_new, num_y_new) sets new shape
            num_threads: number of threads used (degree of parallelism)
            order: the order of interpolation (<= 5)
                large value can improve precision, but also slows down the computaion



        example: Batch = Batch.resize(num_slices_new=128, num_x_new=256,
                                      num_y_new=256, num_threads=25)
        """

        # save the result into result_stacked
        result_stacked = np.zeros((len(self) *
                                   num_slices_new, num_x_new, num_y_new))

        # define array of args
        args = []
        for num_pat in range(len(self)):

            args_dict = {'chunk': self._data,
                         'start_from': self._lower_bounds[num_pat],
                         'end_from': self._upper_bounds[num_pat],
                         'num_slices_new': num_slices_new,
                         'num_x_new': num_x_new,
                         'num_y_new': num_y_new,
                         'res': result_stacked,
                         'start_to': num_pat * num_slices_new}

            args.append(args_dict)

        # print(args)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for arg in args:
                executor.submit(resize_patient_numba, **arg)

        # add info to history
        info = {}
        info['method'] = 'resize'
        info['params'] = {'num_slices_new': num_slices_new,
                          'num_x_new': num_x_new,
                          'num_y_new': num_y_new,
                          'num_threads': num_threads,
                          'order': order}

        self.history.append(info)

        # change data
        self._data = result_stacked

        # change lower/upper bounds
        cur_pat_num = len(self)
        self._lower_bounds = np.arange(cur_pat_num) * num_slices_new
        self._upper_bounds = np.arange(1, cur_pat_num + 1) * num_slices_new

        return self

    def get_mask(self, erosion_radius=7, num_threads=8):
        """
        multithreaded
        computation of lungs' segmentating mask
        args:
            -erosion_radius: radius for erosion of lungs' border


        remember, our patient version of segmentaion has signature
            get_mask_patient(chunk, start_from, end_from, res,
                                      start_to, erosion_radius = 7):

        """
        # we put mask into array
        result_stacked = np.zeros_like(self._data)

        # define array of args
        args = []
        for num_pat in range(len(self)):

            args_dict = {'chunk': self._data,
                         'start_from': self._lower_bounds[num_pat],
                         'end_from': self._upper_bounds[num_pat],
                         'res': result_stacked,
                         'start_to': self._lower_bounds[num_pat],
                         'erosion_radius': erosion_radius}
            args.append(args_dict)

        # run threads and put the fllter into result_stacked
        # print(args)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for arg in args:
                executor.submit(get_mask_patient, **arg)

        # return mask
        return result_stacked

    @action
    def segment(self, erosion_radius=2, num_threads=8):
        """
        lungs segmenting function
            changes self

        sets hu of every pixes outside lungs
            to DARK_HU

        example:
            batch = batch.segment(erosion_radius=4, num_threads=20)
        """
        # get mask with specified params
        # reverse it and set not-lungs to DARK_HU

        lungs = self.get_mask(erosion_radius=erosion_radius,
                              num_threads=num_threads)
        self._data = self._data * lungs

        result_mask = 1 - lungs
        result_mask *= DARK_HU

        # apply mask to self.data
        self._data += result_mask

        # add info about segmentation to history
        info = {}
        info['method'] = 'segmentation'

        info['params'] = {'erosion_radius': erosion_radius,
                          'num_threads': num_threads}

        self.history.append(info)

        return self

    def get_axial_slice(self, person_number, slice_height):
        """
        get axial slice (e.g., for plots)

        args: person_number - can be either
            number of person in the batch
            or index of the person
                whose axial slice we need

        slice_height: e.g. 0.7 means that we take slice with number
            int(0.7 * number of slices for person)

        example: patch = batch.get_axial_slice(5, 0.6)
                 patch = batch.get_axial_slice(self.index[5], 0.6)
                 # here self.index[5] usually smth like 'a1de03fz29kf6h2'

        """
        margin = int(slice_height * self[person_number].shape[0])

        patch = self[person_number][margin, :, :]
        return patch

    @action
    def dump(self, dst, fmt='blosc'):
        """
        dump on specified path and format
            create folder corresponding to each patient

        example:
            # initialize batch and load data
            ind = ['1ae34g90', '3hf82s76', '2ds38d04']
            batch = BatchCt(ind)

            batch.load(...)

            batch.dump('./data/blosc_preprocessed')
            # the command above creates files

            # ./data/blosc_preprocessed/1ae34g90/data.blk
            # ./data/blosc_preprocessed/3hf82s76/data.blk
            # ./data/blosc_preprocessed/2ds38d04/data.blk
        """
        if fmt != 'blosc':
            raise NotImplementedError(
                'Dump to {} not implemented yet'.format(fmt))

        for patient in self.indices:
            # view on patient data
            pat_data = self[patient]
            # pack the data
            packed = blosc.pack_array(pat_data, cname='zstd', clevel=1)

            # remove directory if exists
            if os.path.exists(os.path.join(dst, patient)):
                shutil.rmtree(os.path.join(dst, patient))

            # put blosc on disk
            os.makedirs(os.path.join(dst, patient))

            with open(os.path.join(dst, patient, 'data.blk'), mode='wb') as file:
                file.write(packed)

        # add info in self.history
        info = {}
        info['method'] = 'dump'
        info['params'] = {'path': dst}
        self.history.append(info)

        return self
