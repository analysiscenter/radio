# pylint: disable=no-name-in-module, undefined-variable
""" contains Batch class for storing Ct-scans """

import os
import shutil

import numpy as np
import aiofiles
import blosc
import dicom
import SimpleITK as sitk

from .dataset_import import Batch, action, inbatch_parallel, any_action_failed

from .resize import resize_patient_numba
from .segment import calc_lung_mask_numba
from .mip import xip_fn_numba
from .flip import flip_patient_numba
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
        3. _bounds: 1d-array of bound-floors for each patient
            has length = number of patients + 1


    Important methods:
        1. __init__(self, index):
            basic initialization of patient
            in accordance with Batch.__init__
            given base class Batch

        2. load(self, src, fmt, upper_bounds):
            builds skyscraper of patients
            from either 'dicom'|'raw'|'blosc'|'ndarray'
            returns self

        2. resize(self, num_x_new, num_y_new, num_slices_new,
                  order, num_threads):
            transform the shape of all patients to new_sizes
            method is spline iterpolation(order = order)
            the function is multithreaded in num_threads
            returns self

        3. dump(self, dst)
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

        6. flip(self)
            invert slices corresponding to each patient
            do not change the order of patients
            changes self, returns self

        7. normalize_hu(self, min_hu=-1000, max_hu=400):
            normalizes hu-densities to interval [0, 255]
            trims hus outside range [min_hu, max_hu]
            then scales to [0, 255]
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
        self._bounds = None
        self.origin = None
        self.spacing = None

        self._init_data()

        self._crop_centers = np.array([], dtype=np.int32)
        self._crop_sizes = np.array([], dtype=np.int32)

    def _init_data(self, source=None, bounds=None, origin=None, spacing=None):
        """Initialize _data, _bounds, _crop_centers, _crop_sizes atteributes.

        This method is called inside __init__ and some other methods
        and used as initializer of batch inner structures.

        Args:
        - source: ndarray(n_patients * k, l, m) or None(default) that will be used
        as self._data;
        - bounds: ndarray(n_patients, dtype=np.int) or None(default) that will be
        uses as self._bounds;
        - origin: ndarray(n_patients, 3) or None(default) that will be used as
        self.origin attribute representing patients origins in world coordinate
        system. None value will be converted to zero-array.
        - spacing: ndarray(n_patients, 3) or None(default) that will be used as
        self.spacing attribute representing patients spacings in world coordinate
        system. None value will be converted to ones-array.
        """
        # pylint: disable=attribute-defined-outside-init
        self._data = source
        self._bounds = bounds if bounds is not None else np.array([], dtype='int')
        self.origin = origin if origin is not None else np.zeros((len(self), 3))
        self.spacing = spacing if spacing is not None else np.ones((len(self), 3))

    @action
    def load(self, fmt='dicom', source=None,
             bounds=None, origin=None, spacing=None):    # pylint: disable=arguments-differ
        """
        builds batch of patients

        args:
            source - source array with skyscraper, needed iff fmt = 'ndarray'
            bounds - bound floors for patients
            fmt - type of data.
                Can be 'dicom'|'blosc'|'raw'|'ndarray'

        Dicom example:

            # initialize batch for storing batch of 3 patients
            # with following IDs
            index = FilesIndex(path="/some/path/*.dcm", no_ext=True)
            batch = CTImagesBatch(index)
            batch.load(fmt='dicom')

        Ndarray example:
            # source_array stores a batch (concatted 3d-scans, skyscraper)
            # say, ndarray with shape (400, 256, 256)

            # bounds stores ndarray of last floors for each patient
            # say, source_ubounds = np.asarray([0, 100, 400])
            batch.load(src=source_array, fmt='ndarray', bounds=bounds)

        """
        # if ndarray. Might be better to put this into separate function
        if fmt == 'ndarray':
            self._init_data(source, bounds, origin, spacing)
        elif fmt == 'dicom':
            self._load_dicom()              # pylint: disable=no-value-for-parameter
        elif fmt == 'blosc':
            self._load_blosc()              # pylint: disable=no-value-for-parameter
        elif fmt == 'raw':
            self._load_raw()                # pylint: disable=no-value-for-parameter
        else:
            raise TypeError("Incorrect type of batch source")
        return self


    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_dicom(self, patient_id, *args, **kwargs):                # pylint: disable=unused-argument
        """
        read, prepare and put stacked 3d-scans in an array
            return the array

        args:
            patient_id - index of patient from batch, whose scans we need to
            stack

        Important operations performed here:
         - conversion to hu using meta from dicom-scans
        """
        patient_folder = self.index.get_fullpath(patient_id)

        list_of_dicoms = [dicom.read_file(os.path.join(patient_folder, s)) for s in os.listdir(patient_folder)]

        list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)
        intercept_pat = list_of_dicoms[0].RescaleIntercept
        slope_pat = list_of_dicoms[0].RescaleSlope

        patient_data = np.stack([s.pixel_array for s in list_of_dicoms]).astype(np.int16)

        patient_data[patient_data == AIR_HU] = 0

        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)
        return patient_data

    @inbatch_parallel(init='indices', post='_post_default', target='async')
    async def _load_blosc(self, patient_id, *args, **kwargs):                # pylint: disable=unused-argument
        """
        read, prepare and put 3d-scans in array from blosc
            return the array

        args:
            patient_id - index of patient from batch, whose scans we need to
            stack

            *no conversion to hu here
        """
        blosc_dir_path = os.path.join(self.index.get_fullpath(patient_id), 'data.blk')
        async with aiofiles.open(blosc_dir_path, mode='rb') as file:
            packed = await file.read()
        return blosc.unpack_array(packed)

    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_raw(self, patient_id, *args, **kwargs):  # pylint: disable=unused-argument
        """Read, prepare and put 3d-scans in array from raw(mhd).

        This method reads 3d-scans from mhd format
        in CTImagesMaskedBatch object. This method additionaly
        initializes origin and spacing attributes.

        Args:
        - patient_id: index of patient from batch, whose scans need to
        be put in stack(skyscraper);

        Return :
        - ndarray(Nz, Ny, Nx) patient's data array;
        """
        raw_data = sitk.ReadImage(self.index.get_fullpath(patient_id))
        patient_pos = self.index.get_pos(patient_id)
        self.origin[patient_pos, :] = np.array(raw_data.GetOrigin())[::-1]
        self.spacing[patient_pos, :] = np.array(raw_data.GetSpacing())[::-1]
        return sitk.GetArrayFromImage(raw_data)

    @action
    @inbatch_parallel(init='indices', post='_post_default', target='async', update=False)
    async def dump(self, patient, dst, fmt='blosc'):
        """
        dump on specified path and format
            create folder corresponding to each patient

        example:
            # initialize batch and load data
            ind = ['1ae34g90', '3hf82s76', '2ds38d04']
            batch = BatchCt(ind)
            batch.load(...)
            batch.dump(dst='./data/blosc_preprocessed')
            # the command above creates files

            # ./data/blosc_preprocessed/1ae34g90/data.blk
            # ./data/blosc_preprocessed/3hf82s76/data.blk
            # ./data/blosc_preprocessed/2ds38d04/data.blk
        """
        if fmt != 'blosc':
            raise NotImplementedError('Dump to {} is not implemented yet'.format(fmt))

        # view on patient data
        pat_data = self.get_image(patient)
        # pack the data
        packed = blosc.pack_array(pat_data, cname='zstd', clevel=1)

        # remove directory if exists
        if os.path.exists(os.path.join(dst, patient)):
            shutil.rmtree(os.path.join(dst, patient))

        # put blosc on disk
        os.makedirs(os.path.join(dst, patient))
        async with aiofiles.open(os.path.join(dst, patient, 'data.blk'), mode='wb') as file:
            _ = await file.write(packed)

        return None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        indexation of patients by []

        args:
            self
            index - can be either number (int) of patient
                         in self from [0,..,len(self.index) - 1]
                    or index from self.index
        """
        return self.get_image(index)

    def get_image(self, index):
        """
        get view on patient data

        args:
            index - can be either position of patient in self._data
                or index from self.index
        """
        if isinstance(index, int):
            if index < self._bounds.shape[0] - 1 and index >= 0:
                pos = index
            else:
                raise IndexError("Index is out of range")
        else:
            pos = self.index.get_pos(index)

        lower = self._bounds[pos]
        upper = self._bounds[pos + 1]
        return self._data[lower:upper, :, :]

    @property
    def shape(self):
        """Get CTImages shapes for all patients in CTImagesBatch.

        This property returns ndarray(n_patients, 3) containing
        shapes of data for each patient(first dimension).
        """
        shapes = np.zeros((len(self), 3))
        shapes[:, 0] = self.upper_bounds - self.lower_bounds
        shapes[:, 1], shapes[:, 2] = self.get_image(0).shape[1:]
        return shapes

    @property
    def lower_bounds(self):
        """Get lower bounds of patients data in CTImagesBatch.

        This property returns ndarray(n_patients,) containing
        lower bounds of patients data along z-axis.
        """
        return self._bounds[:-1]

    @property
    def upper_bounds(self):
        """Get upper bounds of patients data in CTImagesBatch.

        This property returns ndarray(n_patients,) containing
        upper bounds of patients data along z-axis.
        """
        return self._bounds[1:]

    @property
    def slice_shape(self):
        """Get shape of slice in yx-plane.

        This property returns ndarray(2,) containing shape of scan slice
        in yx-plane.
        """
        return np.asarray(self._data.shape[1:], np.int)

    def rescale_spacing(self, new_shape):
        """Rescale patients' spacing parameter after resise.

        This method recomputes spacing values
        for patients' data stored in CTImagesBatch after resize operation.

        Args:
        - new_shape: new shape of single patient data array, supposed to be
        np.array([j, k, l], dtype=np.int) where j, k, l -- sizes of
        single patient data array along z, y, x correspondingly;
        Return:
        - new_spacing: ndarray(n_patients, 3) with spacing values for each
        patient along z, y, x axes.
        """
        return (self.spacing * self.shape) / new_shape

    def _post_default(self, list_of_arrs, update=True, new_batch=False, **kwargs):    # pylint: disable=unused-argument
        """
        gatherer of outputs of different workers
            assumes that output of each worker corresponds to patient data
        """
        if any_action_failed(list_of_arrs):
            raise ValueError("Failed while parallelizing")

        res = self
        if update:
            new_data = np.concatenate(list_of_arrs, axis=0)
            new_bounds = np.cumsum(np.array([len(a) for a in [[]] + list_of_arrs]))
            if new_batch:
                batch = type(self)(self.index)
                batch.load(fmt='ndarray', src=new_data, bounds=new_bounds,
                           origin=self.origin, spacing=self.spacing)
                res = batch
            else:
                self._data = new_data
                self._bounds = new_bounds
        return res

    def _init_images(self, **kwargs):               # pylint: disable=unused-argument
        return [self.get_image(patient_id) for patient_id in self.indices]

    def _post_crop(self, list_of_arrs, **kwargs):   # pylint: disable=unused-argument
        # TODO: check for errors
        crop_array = np.array(list_of_arrs)
        self._crop_centers = crop_array[:, :, 2]
        self._crop_sizes = crop_array[:, :, : 2]


    def _init_rebuild(self, **kwargs):
        """
        args-fetcher for parallelization using decorator
            can be used when batch-data is rebuild from scratch
        if shape is supplied as one of the args
            assumes that data should be resizd
        """
        if 'shape' in kwargs:
            num_slices, y, x = kwargs['shape']
            new_bounds = num_slices * np.arange(len(self) + 1)
            new_data = np.zeros((num_slices * len(self), y, x))
        else:
            new_bounds = self._bounds
            new_data = np.zeros_like(self._data)

        all_args = []
        for i in range(len(self.indices)):
            out_patient = new_data[new_bounds[i] : new_bounds[i + 1], :, :]
            item_args = {'patient': self.get_image(i),
                         'out_patient': out_patient,
                         'res': new_data}

            all_args += [item_args]

        return all_args

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):   # pylint: disable=unused-argument
        """
        gatherer of outputs from different workers for
            ops, requiring complete rebuild of batch._data
        args:
            new_batch: if True, returns new batch with data
                agregated from workers_ouputs
        """
        if any_action_failed(all_outputs):
            raise ValueError("Failed while parallelizing")

        new_bounds = np.cumsum([patient_shape[0] for _, patient_shape
                                in [[0, (0, )]] + all_outputs])
        # each worker returns the same ref to the whole res array
        new_data, _ = all_outputs[0]

        if 'shape' in kwargs:
            new_spacing = self.rescale_spacing(kwargs['shape'])
        else:
            new_spacing = self.spacing

        if new_batch:
            batch_res = type(self)(self.index)
            batch_res.load(source=new_data, bounds=new_bounds, fmt='ndarray',
                           origin=self.origin, spacing=new_spacing)
            return batch_res
        else:
            self._init_data(source=new_data, bounds=new_bounds,
                            origin=self.origin, spacing=new_spacing)
            return self

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


    @inbatch_parallel(init='_init_images', post='_post_crop', target='nogil')
    def _crop_params_patients(self, *args, **kwargs):                    # pylint: disable=unused-argument,no-self-use
        """
        calculate params for crop, calling return_black_border_array
        """
        return rbba

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='nogil')
    def resize(self, shape=(128, 256, 256), order=3, *args, **kwargs):    # pylint: disable=unused-argument, no-self-use
        """
        performs resize (change of shape) of each CT-scan in the batch.
            When called from Batch, changes Batch
            returns self
        args:
            shape: needed shape after resize in order z, y, x
                *note that the order of axes in data is z, y, x
                 that is, new patient shape = (shape[0], shape[1], shape[2])
            n_workers: number of threads used (degree of parallelism)
                *note: available in the result of decoration of the function
                above
            order: the order of interpolation (<= 5)
                large value improves precision, but slows down the computaion
        example:
            shape = (128, 256, 256)
            Batch = Batch.resize(shape=shape, n_workers=20, order=2)
        """
        return resize_patient_numba


    @action
    @inbatch_parallel(init='_init_images', post='_post_default', target='nogil', new_batch=True)
    def make_xip(self, step=2, depth=10, func='max', projection='axial', *args, **kwargs):    # pylint: disable=unused-argument, no-self-use
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
        return xip_fn_numba(func, projection, step, depth)

    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='nogil', new_batch=True)
    def calc_lung_mask(self, *args, **kwargs):     # pylint: disable=unused-argument, no-self-use
        """ Return a mask for lungs """
        return calc_lung_mask_numba

    @action
    def segment(self, erosion_radius=2):
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

        mask_batch = self.calc_lung_mask(erosion_radius=erosion_radius)
        lungs_mask = mask_batch.data
        self._data = self._data * lungs_mask

        result_mask = 1 - lungs_mask
        result_mask *= DARK_HU

        self._data += result_mask

        return self

    @action
    def normalize_hu(self, min_hu=-1000, max_hu=400):
        """
        normalizes hu-densities to interval [0, 255]:
            trims hus outside range [min_hu, max_hu]
            then scales to [0, 255]

        example:
            batch = batch.normalize_hu(min_hu=-1300, max_hu=600)
        """

        # trimming and scaling to [0, 1]
        self._data = (self._data - min_hu) / (max_hu - min_hu)
        self._data[self._data > 1] = 1.
        self._data[self._data < 0] = 0.

        # scaling to [0, 255]
        self._data *= 255
        return self

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='nogil')
    def flip(self):    # pylint: disable=no-self-use
        """
        flip each patient
            i.e. invert the order of slices for each patient
            does not change the order of patients
            changes self

        example:
            batch = batch.flip()
        """
        return flip_patient_numba

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
