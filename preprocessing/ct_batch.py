# pylint: disable=too-many-arguments
# pylint: disable=undefined-variable
# pylint: disable=no-member
""" contains Batch class for storing Ct-scans """

import os
import cloudpickle

import numpy as np
import aiofiles
import blosc
import dicom
import SimpleITK as sitk

from ..dataset import Batch, action, inbatch_parallel, any_action_failed, DatasetIndex

from .resize import resize_scipy, resize_pil
from .segment import calc_lung_mask_numba
from .mip import xip_fn_numba
from .flip import flip_patient_numba
from .crop import return_black_border_array as rbba
from .crop import make_central_crop
from .patches import get_patches_numba, assemble_patches, calc_padding_size
from .rotate import rotate_3D, random_rotate_3D


AIR_HU = -2000
DARK_HU = -2000


class CTImagesBatch(Batch):  # pylint: disable=too-many-public-methods

    """
    class for storing batch of CT(computed tomography) 3d-scans.
    This class is derived from base class Batch defined in dataset submodule.


    Attrs:
        1. index: array of itemsIDs. Usually, itemsIDs are strings
        2. images: 3d-array of stacked scans along number_of_slices axis ("skyscraper")
        3. _bounds: 1d-array of bound-floors for each scan,
            has length = number of items in batch + 1
        4. spacing: 2d-array [number of items X 3] that contains spacing
            of each item (scan) along each axis
        5. origin: 2d-array [number of items X 3] that contains spacing
            of each item along each axis


    Important methods:
        1. __init__(self, index):
            basic initialization of images batch
            in accordance with Batch.__init__
            given base class Batch

        2. load(self, source, fmt, upper_bounds, src_blosc):

            builds skyscraper of scans
            from either 'dicom'|'raw'|'blosc'|'ndarray'
            returns self

        2. resize(self, shape, order):
            transform the shape of all scans to shape=shape
            method is spline iterpolation(order = order)
            the function is multithreaded
            returns self

        3. unify_spacing(self, spacing, shape):
            resize each scan so that its spacing changed to supplied spacing,
            then crop/pad each scan to supplied shape

        4. dump(self, format, dst)
            create a dump of the batch
            in the path-folder
            returns self

            NOTE: as of 06.07.2017, only format='blosc' is supported

        5. calc_lungs_mask(self, erosion_radius=7)
            returns binary-mask for lungs segmentation
            the larger erosion_radius
            the lesser the resulting lungs will be
            * returns mask, not self

        6. segment(self, erosion_radius=2)
            segments using mask from calc_lungs_mask()
            that is, sets to hu = -2000 of pixels outside mask
            changes self, returns self

            *NOTE: segment can be applied only before normalize_hu
                and to batch that contains scans for whole patients

        7. flip(self)
            invert slices corresponding to each scan
            do not change the order of scans
            changes self, returns self

        7. normalize_hu(self, min_hu=-1000, max_hu=400):
            normalizes hu-densities to interval [0, 255]
            trims hus outside range [min_hu, max_hu]
            then scales to [0, 255]
            changes self, returns self

    """

    def __init__(self, index, *args, **kwargs):
        """
        common part of initialization from all formats:
            -execution of Batch construction
            -initialization of all attrs
            -creation of empty lists and arrays

        Args:
            index: index of type DatasetIndex
        """

        super().__init__(index, *args, **kwargs)

        # init all attrs
        self.images = None
        self._bounds = None
        self.origin = None
        self.spacing = None
        self._init_data()

        self._crop_centers = np.array([], dtype=np.int32)
        self._crop_sizes = np.array([], dtype=np.int32)

    @property
    def components(self):
        """ Components-property. See doc of base batch from dataset for information.
                In short, these are names for components of tuple returned from __getitem__.
        """
        return 'images', 'spacing', 'origin'

    def _init_data(self, source=None, bounds=None, origin=None, spacing=None):
        """Initialize images, _bounds, _crop_centers, _crop_sizes atteributes.

        This method is called inside __init__ and some other methods
        and used as initializer of batch inner structures.

        Args:
        - source: ndarray(n_patients * k, l, m) or None(default) that will be used
        as self.images
        - bounds: ndarray(n_patients, dtype=np.int) or None(default) that will be
        uses as self._bounds;
        - origin: ndarray(n_patients, 3) or None(default) that will be used as
        self.origin attribute representing patients origins in world coordinate
        system. None value will be converted to zero-array.
        - spacing: ndarray(n_patients, 3) or None(default) that will be used as
        self.spacing attribute representing patients spacings in world coordinate
        system. None value will be converted to ones-array.
        """
        self.images = source
        self._bounds = bounds if bounds is not None else np.array([], dtype='int')
        self.origin = origin if origin is not None else np.zeros((len(self), 3))
        self.spacing = spacing if spacing is not None else np.ones((len(self), 3))

    @classmethod
    def split(cls, batch, batch_size):
        """ Split batch in two batches of lens=(batch_size, len(batch) - batch_size)

        Args:
            batch: batch to be splitted
            batch_size: len of first half. If batch_size >= len(batch), return None instead
                of a second batch

        Return:
            (first_half, second_half)

        NOTE: the method does not change the structure of batch.index. Indices of created
            batches are simply subsets of batch.index.
        """
        if batch_size == 0:
            return (None, batch)

        if batch_size >= len(batch):
            return (batch, None)

        # form indices for both batches
        size_first, _ = batch_size, len(batch) - batch_size
        ix_first = batch.index.create_subset(batch.indices[:size_first])
        ix_second = batch.index.create_subset(batch.indices[size_first:])

        # init batches
        batches = cls(ix_first), cls(ix_second)

        # put non-None components in batch-parts
        for batch_part in batches:
            for component in batch.components:
                if getattr(batch, component) is not None:
                    comps = []
                    for ix in batch_part.indices:
                        # get component for a specific item defined by ix and put into the list
                        comp_pos = batch.get_pos(None, component, ix)
                        comp = getattr(batch, component)[comp_pos]
                        comps.append(comp)

                    # set the component for the whole batch-part
                    source = np.concatenate(comps)
                    setattr(batch_part, component, source)
                else:
                    setattr(batch_part, component, None)

        # set _bounds attrs if filled in batch
        if len(batch._bounds) >= 2:  # pylint: disable=protected-access
            for batch_part in batches:
                n_slices = []
                for ix in batch_part.indices:
                    ix_pos_initial = batch.index.get_pos(ix)
                    n_slices.append(batch.upper_bounds[ix_pos_initial]
                                    - batch.lower_bounds[ix_pos_initial])

                # update _bounds in new batches
                batch_part._bounds = np.cumsum([0] + n_slices, dtype=np.int)  # pylint: disable=protected-access

        return batches

    @classmethod
    def concat(cls, batches):
        """ Concatenate several batches in one large batch. Assume that
                the same components are filled in all supplied batches.

        Args:
            batches: sequence of batches to be concatenated

        Return:
            large batch with len = sum of lens of batches

        NOTE: batches' index is dropped. New large batch has new np-arange
            index
        NOTE: None-entries or batches of len=0 can be included in the list of batches.
            They are simply dropped
        """
        # leave only non-empty batches
        batches = [batch for batch in batches if batch is not None]
        batches = [batch for batch in batches if len(batch) > 0]

        if len(batches) == 0:
            return None

        # create index for the large batch and init batch
        ixbatch = DatasetIndex(np.arange(np.sum([len(batch) for batch in batches])))
        large_batch = cls(ixbatch)

        # set non-none components in the large batch
        for component in batches[0].components:
            comps = None
            if getattr(batches[0], component) is not None:
                comps = np.concatenate([getattr(batch, component) for batch in batches])
            setattr(large_batch, component, comps)

        # set _bounds-attr in large batch
        n_slices = np.zeros(shape=len(large_batch))
        ctr = 0
        for batch in batches:
            n_slices[ctr: ctr + len(batch)] = batch.upper_bounds - batch.lower_bounds
            ctr += len(batch)

        large_batch._bounds = np.cumsum(np.insert(n_slices, 0, 0), dtype=np.int)  # pylint: disable=protected-access
        return large_batch

    @classmethod
    def merge(cls, batches, batch_size=None):
        """ Concatenate list of batches and then split the result in two batches of sizes
                (batch_size, sum(lens of batches) - batch_size)

        Args:
            batches: list of batches
            batch_size: size of first resulting batch

        Return:
            (new_batch, rest_batch)

        NOTE: we perform split(of middle-batch) and then two concats
        because of speed considerations;
        even though the code is slightly lengthier
        than it'd be if the order was concat->split.
        """
        if np.sum([len(batch) for batch in batches]) <= batch_size:
            return (cls.concat(batches), None)

        # find a batch that needs to be splitted (middle batch)
        cum_len = 0
        middle = None
        middle_pos = None
        for pos, batch in enumerate(batches):
            cum_len += len(batch)
            if cum_len >= batch_size:
                middle = batch
                middle_pos = pos
                break

        # split middle batch
        left_middle, right_middle = cls.split(middle, len(middle) - cum_len + batch_size)

        # form merged and rest-batches
        merged = cls.concat(batches[:middle_pos] + [left_middle])
        rest = cls.concat([right_middle] + batches[middle_pos + 1:])

        return merged, rest

    @action
    def load(self, fmt='dicom', source=None, bounds=None,  # pylint: disable=arguments-differ
             origin=None, spacing=None, src_blosc=None):
        """ Loads 3d scans-data in batch

        Args:
            fmt: type of data. Can be 'dicom'|'blosc'|'raw'|'ndarray'
            source: source array with skyscraper, needed iff fmt = 'ndarray'
            bounds: bound floors for patients. Needed iff fmt='ndarray'
            origin: ndarray [len(bounds) X 3] with world coords of each patient's
                starting pixels. Needed only if fmt='ndarray'
            spacing: ndarray [len(bounds) X 3] with spacings of patients.
                Needed only if fmt='ndarray'
            src_blosc: list/tuple/string with component(s) of batch
                that should be loaded from blosc.
                Needed only if fmt='blosc'. If None, all components are loaded.

        Return:
            self

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
            batch.load(source=source_array, fmt='ndarray', bounds=bounds)

        """
        # if ndarray
        if fmt == 'ndarray':
            self._init_data(source, bounds, origin, spacing)
        elif fmt == 'dicom':
            self._load_dicom()              # pylint: disable=no-value-for-parameter
        elif fmt == 'blosc':
            src_blosc = self.components if src_blosc is None else src_blosc
            # convert src_blosc to iterable 1d-array
            src_blosc = np.asarray(src_blosc).reshape(-1)
            self._load_blosc(src=src_blosc)              # pylint: disable=no-value-for-parameter
        elif fmt == 'raw':
            self._load_raw()                # pylint: disable=no-value-for-parameter
        else:
            raise TypeError("Incorrect type of batch source")
        return self

    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_dicom(self, patient_id, *args, **kwargs):
        """
        Read, prepare and put stacked 3d-scans in an array
            return the array

        args:
            patient_id - index of patient from batch, whose scans we need to
            stack

        Important operations performed here:
         - conversion to hu using meta from dicom-scans
        """
        # put 2d-scans for each patient in a list
        patient_folder = self.index.get_fullpath(patient_id)
        list_of_dicoms = [dicom.read_file(os.path.join(patient_folder, s))
                          for s in os.listdir(patient_folder)]

        list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)
        intercept_pat = list_of_dicoms[0].RescaleIntercept
        slope_pat = list_of_dicoms[0].RescaleSlope

        patient_data = np.stack([s.pixel_array for s in list_of_dicoms]).astype(np.int16)

        patient_data[patient_data == AIR_HU] = 0

        # perform conversion to HU
        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)
        return patient_data

    def _preload_shapes(self):
        """ Read shapes of scans dumped with blosc, update
                self._bounds. The method is used in _init_load_blosc.

            Args:
                __
            Return:
                y, x - components of scan-shape
        """
        shapes = np.zeros((len(self), 3), dtype=np.int)
        for ix in self.indices:
            filename = os.path.join(self.index.get_fullpath(ix), 'images_shape.cpkl')
            ix_pos = self._get_verified_pos(ix)

            # read shape and put it into shapes
            with open(filename, 'rb') as file:
                shapes[ix_pos, :] = cloudpickle.load(file)

        # update bounds of items
        self._bounds = np.cumsum(np.insert(shapes[:, 0], 0, 0), dtype=np.int)

        # return shape of slices
        return shapes[0, 1], shapes[0, 2]

    def _init_load_blosc(self, **kwargs):
        """ Init-func for load from blosc.

        Args:
            src: iterable of components that need to be loaded
        Return
            list of ids of batch-items
        """
        # set images-component to 3d-array of zeroes if the component is to be updated
        if 'images' in kwargs['src']:
            slice_shape = self._preload_shapes()
            skysc_shape = (self._bounds[-1], ) + slice_shape
            self.images = np.zeros(skysc_shape)

        return self.indices

    @inbatch_parallel(init='_init_load_blosc', post='_post_default', target='async', update=False)
    async def _load_blosc(self, patient_id, *args, **kwargs):
        """
        Read, prepare scans from blosc and put them into the right place in the
            skyscraper

        Args:
            patient_id: index of patient from batch, whose scans we need to
                stack
            src: components of data that should be loaded into self,
                1d-array

            *no conversion to hu here
        """

        for source in kwargs['src']:
            # set correct extension for each component and choose a tool
            # for debyting it
            if source in ['spacing', 'origin']:
                ext = '.cpkl'
                unpacker = cloudpickle.loads
            else:
                ext = '.blk'
                unpacker = blosc.unpack_array

            comp_path = os.path.join(self.index.get_fullpath(patient_id), source + ext)

            # read the component
            async with aiofiles.open(comp_path, mode='rb') as file:
                byted = await file.read()

            # de-byte it with the chosen tool
            component = unpacker(byted)

            # update needed slice(s) of component
            comp_pos = self.get_pos(None, source, patient_id)
            getattr(self, source)[comp_pos] = component

        return None

    def _load_raw(self, **kwargs):        # pylint: disable=unused-argument
        """ Load scans from .raw (.mhd)

            *NOTE1: no conversion to hu here (assume densities are already in hu)
            *NOTE2: no multithreading here, as SimpleITK (sitk)-library does not seem
                to work correcly with multithreading
        """
        list_of_arrs = []
        for patient_id in self.indices:
            raw_data = sitk.ReadImage(self.index.get_fullpath(patient_id))
            patient_pos = self.index.get_pos(patient_id)
            list_of_arrs.append(sitk.GetArrayFromImage(raw_data))

            # *.mhd files contain information about scans' origin and spacing;
            # however the order of axes there is inversed:
            # so, we just need to reverse arrays with spacing and origin.
            self.origin[patient_pos, :] = np.array(raw_data.GetOrigin())[::-1]
            self.spacing[patient_pos, :] = np.array(raw_data.GetSpacing())[::-1]

        new_data = np.concatenate(list_of_arrs, axis=0)
        new_bounds = np.cumsum(np.array([len(a) for a in [[]] + list_of_arrs]))
        self.images = new_data
        self._bounds = new_bounds
        return self

    @staticmethod
    async def dump_data(data_items, folder):
        """ Dump data that is contained in data_items on disk in
                specified folder

        Args:
            data_items: dict of data items for dump in form item_name.ext: item
                (e.g.: {'images.blk': scans, 'mask.blk': mask, 'spacing.cpkl': spacing})
            folder: folder to dump data-items in

        Return:
            ____

        *NOTE: depending on supplied format, each data-item will be either
            cloudpickle-serialized (if .cpkl) or blosc-packed (if .blk)
        """

        # create directory if does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # infer extension of each item, serialize/blosc-pack and dump the item
        for filename, data in data_items.items():
            ext = filename.split('.')[-1]
            if ext == 'blk':
                byted = blosc.pack_array(data, cname='zstd', clevel=1)
            elif ext == 'cpkl':
                byted = cloudpickle.dumps(data)
            async with aiofiles.open(os.path.join(folder, filename),
                                     mode='wb') as file:
                _ = await file.write(byted)

        return None

    @action
    @inbatch_parallel(init='indices', post='_post_default', target='async', update=False)
    async def dump(self, patient, dst, src=None, fmt='blosc'):
        """ Dump scans data (3d-array) on specified path in specified format

        Args:
            dst: general folder in which all patients' data should be put
            src: component(s) that we need to dump (smth iterable or string). If not
                supplied, dump all components + shapes of scans
            fmt: format of dump. Currently only blosc-format is supported;
                in this case folder for each patient is created, patient's data
                is put into images.blk, attributes are put into files attr_name.cpkl
                (e.g., spacing.cpkl)

        example:
            # initialize batch and load data
            ind = ['1ae34g90', '3hf82s76']
            batch = CTImagesBatch(ind)
            batch.load(...)
            batch.dump(dst='./data/blosc_preprocessed')
            # the command above creates files

            # ./data/blosc_preprocessed/1ae34g90/images.blk
            # ./data/blosc_preprocessed/1ae34g90/spacing.cpkl
            # ./data/blosc_preprocessed/1ae34g90/origin.cpkl
            # ./data/blosc_preprocessed/1ae34g90/shape.cpkl

            # ./data/blosc_preprocessed/3hf82s76/images.blk
            # ./data/blosc_preprocessed/1ae34g90/spacing.cpkl
            # ./data/blosc_preprocessed/3hf82s76/origin.cpkl
            # ./data/blosc_preprocessed/1ae34g90/shape.cpkl
        """
        # if src is not supplied, dump all components and shapes
        if src is None:
            src = self.components + ('images_shape', )

        if fmt != 'blosc':
            raise NotImplementedError('Dump to {} is not implemented yet'.format(fmt))

        # convert src to iterable 1d-array
        src = np.asarray(src).reshape(-1)
        data_items = dict()

        # whenever images are to be dumped, shape should also be dumped
        if 'images' in src and 'images_shape' not in src:
            src = tuple(src) + ('images_shape', )

        # set correct extension to each component and add it to items-dict
        for source in list(src):
            if source in ['spacing', 'origin', 'images_shape']:
                ext = '.cpkl'
            else:
                ext = '.blk'
            # determine position in data of source-component for the patient
            comp_pos = self.get_pos(None, source, patient)
            data_items.update({source + ext: getattr(self, source)[comp_pos]})

        # set patient-specific folder
        folder = os.path.join(dst, patient)

        return await self.dump_data(data_items, folder)

    def get_pos(self, data, component, index):
        """ Return a posiiton of a component in data for a given index

        *NOTE: this is an overload of get_pos from base Batch-class,
            see corresponding docstring for detailed explanation.
        """
        if data is None:
            ind_pos = self._get_verified_pos(index)
            if component == 'images':
                return slice(self.lower_bounds[ind_pos], self.upper_bounds[ind_pos])
            else:
                return slice(ind_pos, ind_pos + 1)
        else:
            return index

    def _get_verified_pos(self, index):
        """Get verified position of patient in batch by index.

        Firstly, check if index is instance of str or int. If int
        then it is supposed that index represents patient's position in Batch.
        If fetched position is out of bounds then Exception is generated.
        If str then position of patient is fetched.

        Args:
            index: can be either position of patient in self.images
                or index from self.index

        Return:
            if supplied index is int, return supplied number,
            o/w return the position of patient with supplied index
        """
        if isinstance(index, int):
            if index < len(self) and index >= 0:
                pos = index
            else:
                raise IndexError("Index is out of range")
        else:
            pos = self.index.get_pos(index)
        return pos

    @property
    def images_shape(self):
        """Get CTImages shapes for all patients in CTImagesBatch.

        This property returns ndarray(n_patients, 3) containing
        shapes of data for each patient(first dimension).
        """
        shapes = np.zeros((len(self), 3), dtype=np.int)
        shapes[:, 0] = self.upper_bounds - self.lower_bounds
        shapes[:, 1], shapes[:, 2] = self.slice_shape
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
        return np.asarray(self.images.shape[1:])

    def rescale(self, new_shape):
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
        return (self.spacing * self.images_shape) / new_shape

    def _post_default(self, list_of_arrs, update=True, new_batch=False, **kwargs):
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
            params = dict(source=new_data, bounds=new_bounds,
                          origin=self.origin, spacing=self.spacing)
            if new_batch:
                batch = type(self)(self.index)
                batch.load(fmt='ndarray', **params)
                res = batch
            else:
                self._init_data(**params)
        return res

    def _post_components(self, list_of_dicts, **kwargs):
        """ Gather outputs of different workers, update self. Assume each
                output corresponds to a scan (one batch item) and is a dict of format
                {component: what_should_be_put_into_component}.

            Return:
                self
        """
        if any_action_failed(list_of_dicts):
            raise ValueError("Failed while parallelizing")

        # if images is in dict, update bounds
        if 'images' in list_of_dicts[0]:
            list_of_images = [worker_res['images'] for worker_res in list_of_dicts]
            new_bounds = np.cumsum(np.array([len(a) for a in [[]] + list_of_images]))
            new_data = np.concatenate(list_of_images, axis=0)
            params = dict(source=new_data, bounds=new_bounds,
                          origin=self.origin, spacing=self.spacing)
            self._init_data(**params)

        # loop over other components that we need to update
        for component in list_of_dicts[0]:
            if component == 'images':
                pass
            else:
                # concatenate comps-outputs for different scans and update self
                list_of_component = [worker_res[component] for worker_res in list_of_dicts]
                new_comp = np.concatenate(list_of_component, axis=0)
                setattr(self, component, new_comp)

        return self

    def _init_images(self, **kwargs):
        return [self.get(patient_id, 'images') for patient_id in self.indices]

    def _post_crop(self, list_of_arrs, **kwargs):
        # TODO: check for errors
        crop_array = np.array(list_of_arrs)
        self._crop_centers = crop_array[:, :, 2]
        self._crop_sizes = crop_array[:, :, : 2]

    def _init_rebuild(self, **kwargs):
        """ Args-fetcher for parallelization using inbatch-parallel, used when 'images'-attr
                is rebuild from scratch.
        Args:
            shape: if supplied, assume that images-component will be of this shape
                in the result of action execution
            spacing: if supplied, assume that unify_spacing is performed

        Return:
            list of arg-dicts for different workers
        """
        if 'shape' in kwargs:
            num_slices, y, x = kwargs['shape']
            new_bounds = num_slices * np.arange(len(self) + 1)
            new_data = np.zeros((num_slices * len(self), y, x))
        else:
            new_bounds = self._bounds
            new_data = np.zeros_like(self.images)

        all_args = []
        for i in range(len(self)):
            out_patient = new_data[new_bounds[i]: new_bounds[i + 1], :, :]
            item_args = {'patient': self.get(i, 'images'),
                         'out_patient': out_patient,
                         'res': new_data}

            # for unify_spacing
            if 'spacing' in kwargs:
                shape_after_resize = (self.images_shape * self.spacing
                                      / np.asarray(kwargs['spacing']))
                shape_after_resize = np.rint(shape_after_resize).astype(np.int)
                item_args['res_factor'] = self.spacing[i, :] / np.array(kwargs['spacing'])
                item_args['shape_resize'] = shape_after_resize[i, :]

            all_args += [item_args]

        return all_args

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):
        """ Gather outputs of different workers for actions, which
                require complete rebuild of images-comp.

        Args:
            all_outputs: list of workers' outputs. Each item is given by tuple
                (ref on new images-comp for whole batch, specific scan's shape)
            new_batch: if True, returns new batch with data agregated
                from all_ouputs. O/w changes self.
        """
        if any_action_failed(all_outputs):
            raise ValueError("Failed while parallelizing")

        new_bounds = np.cumsum([patient_shape[0] for _, patient_shape
                                in [[0, (0, )]] + all_outputs])
        # each worker returns the same ref to the whole res array
        new_data, _ = all_outputs[0]

        # recalculate new_attrs of a batch

        # for resize/unify_spacing: if shape is supplied, assume post
        # is for resize or unify_spacing
        if 'shape' in kwargs:
            new_spacing = self.rescale(kwargs['shape'])
        else:
            new_spacing = self.spacing

        # for unify_spacing: if spacing is supplied, assume post
        # is for unify_spacing
        if 'spacing' in kwargs:
            # recalculate origin, spacing
            shape_after_resize = np.rint(self.images_shape * self.spacing
                                         / np.asarray(kwargs['spacing']))
            overshoot = shape_after_resize - np.asarray(kwargs['shape'])
            new_spacing = self.rescale(new_shape=shape_after_resize)
            new_origin = self.origin + new_spacing * (overshoot // 2)
        else:
            new_origin = self.origin

        # build/update batch with new data and attrs
        params = dict(source=new_data, bounds=new_bounds,
                      origin=new_origin, spacing=new_spacing)
        if new_batch:
            batch_res = type(self)(self.index)
            batch_res.load(fmt='ndarray', **params)
            return batch_res
        else:
            self._init_data(**params)
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
    def _crop_params_patients(self, *args, **kwargs):
        """
        calculate params for crop, calling return_black_border_array
        """
        return rbba

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='threads')
    def resize(self, patient, out_patient, res, shape=(128, 256, 256), method='pil-simd',
               axes_pairs=None, resample=None, order=3, *args, **kwargs):
        """ Resize (change shape of) each CT-scan in the batch.
                When called from a batch, changes this batch.
        Args:
            shape: needed shape after resize in order z, y, x.
                NOTE: the order of axes in images is z, y, x. That is,
                shape of each scan after resize is (shape[0], shape[1], shape[2])
            method: interpolation package to be used. Can be either 'pil-simd'
                or 'scipy'. Pil-simd ensures better quality and speed on configurations
                with average number of cores. On the contrary, scipy is better scaled and
                can show better performance on systems with large number of cores
            axes_pairs: pairs of axes that will be used for performing pil-simd resize.
                If None, set to ((0, 1), (1, 2)). In general, this arg has to be
                a list/tuple of tuples of len=2 (pairs). The more pairs one uses,
                the more precise will be the result (while computation will take more time).
                Min number of pairs to use is 1, while at max there can be 3 * 2 = 6 pairs.
            resample: filter of pil-simd resize. By default set to bilinear. Can be any of filters
                supported by PIL.Image.
            order: the order of scipy-interpolation (<= 5)
                large value improves precision, but slows down the computaion.
        Return:
            self
        example:
            shape = (128, 256, 256)
            Batch = Batch.resize(shape=shape, order=2, method='scipy')
            Bacch = Batch.resize(shape=shape, resample=PIL.Image.BILINEAR)
        """
        if method == 'scipy':
            args_resize = dict(patient=patient, out_patient=out_patient, res=res, order=order)
            return resize_scipy(**args_resize)
        elif method == 'pil-simd':
            args_resize = dict(input_array=patient, output_array=out_patient,
                               res=res, axes_pairs=axes_pairs, resample=resample)
            return resize_pil(**args_resize)

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='threads')
    def unify_spacing(self, patient, out_patient, res, res_factor,
                      shape_resize, spacing=(1, 1, 1), shape=(128, 256, 256),
                      method='pil-simd', order=3, padding='edge', axes_pairs=None,
                      resample=None, *args, **kwargs):
        """ Unify spacing of all patients using resize, then crop/pad resized array
                to supplied shape.
        Args:
            spacing: needed spacing in mm
            shape: needed shape after crop/pad
            method: interpolation package to be used for resize ('pil-simd' | resize). See doc of
                CTImagesBatch.resize for more information
            order: order of scipy-interpolation (<=5)
            padding: mode of padding, any of those supported by np.pad
            axes_pairs: pairs of axes that will be used for performing pil-simd resize
            resample: filter of pil-simd resize

            NOTE: see doc of CTImagesBatch.resize for more info about methods' params.
        Return:
            self
        """
        if method == 'scipy':
            args_resize = dict(patient=patient, out_patient=out_patient,
                               res=res, order=order, res_factor=res_factor, padding=padding)
            return resize_scipy(**args_resize)
        elif method == 'pil-simd':
            args_resize = dict(input_array=patient, output_array=out_patient,
                               res=res, axes_pairs=axes_pairs, resample=resample,
                               shape_resize=shape_resize, padding=padding)
            return resize_pil(**args_resize)

    @action
    @inbatch_parallel(init='_init_images', post='_post_default', target='threads')
    def rotate(self, image, degree, axes=(1, 2), **kwargs):
        """ Rotate 3D images in batch on specific angle in plane.

        Args:
        - degree: float, degree of rotation;
        - axes: tuple(int, int), plane of rotation specified by two axes;

        Returns:
        - ndarray(l, k, m), 3D rotated image;

        *NOTE: zero padding automatically added after rotation;
        """
        return rotate_3D(image, degree, axes)

    @action
    @inbatch_parallel(init='_init_images', post='_post_default', target='threads')
    def random_rotate(self, image, max_degree, axes=(1, 2), **kwargs):
        """ Perform rotation of 3D image in batch on random angle.

        Args:
        - max_degree: float, maximum rotation angle;
        - axes: tuple(int, int), plane of rotation specified by two axes;

        Returns:
        - ndarray(l, k, m), 3D rotated image;

        *NOTE: zero padding automatically added after rotation;
        """
        return random_rotate_3D(image, max_degree, axes)

    @action
    @inbatch_parallel(init='_init_images', post='_post_default', target='nogil', new_batch=True)
    def make_xip(self, step=2, depth=10, func='max', projection='axial', *args, **kwargs):
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
        # get mask with specified params, apply it to scans
        mask_batch = self.calc_lung_mask(erosion_radius=erosion_radius)
        lungs_mask = mask_batch.images
        self.images *= lungs_mask

        # reverse the mask and set not-lungs to DARK_HU
        result_mask = 1 - lungs_mask
        result_mask *= DARK_HU

        self.images += result_mask

        return self

    @action
    def central_crop(self, crop_size, **kwargs):
        """ Make crop with given size from center of images.

        Args:
        - crop_size: tuple(int, int, int), size of crop;
        """
        crop_size = np.asarray(crop_size)
        crop_halfsize = np.rint(crop_size / 2)
        img_shapes = [np.asarray(self.get(i, 'images').shape) for i in range(len(self))]
        if any(np.any(shape < crop_size) for shape in img_shapes):
            raise ValueError("Crop size must be smaller than size of inner 3D images")

        cropped_images = []
        for i in range(len(self)):
            image = self.get(i, 'images')
            cropped_images.append(make_central_crop(image, crop_size))

        self._bounds = np.cumsum([0] + [crop_size[0]] * len(self))
        self.images = np.concatenate(cropped_images, axis=0)
        self.origin = self.origin + self.spacing * crop_halfsize
        return self

    def get_patches(self, patch_shape, stride, padding='edge', data_attr='images'):
        """ Extract patches of size patch_shape with specified
                stride

        Args:
            patch_shape: tuple/list/ndarray of len=3 with needed
                patch shape
            stride: tuple/list/ndarray of len=3 with stride that we
                use to slide over each patient's data
            padding: type of padding (see doc of np.pad for available types)
                say, 3.6 windows of size=patch_shape with stride
                can be exracted from each patient's data.
                Then data will be padded s.t. 4 windows can be extracted
            data_attr: name of attribute where the data is stored
                images by default

        Return:
            4d-ndaray of patches; first dimension enumerates patches
        *NOTE: the shape of all patients is assumed to be the same
        """

        patch_shape, stride = np.asarray(patch_shape), np.asarray(stride)
        img_shape = self.images_shape[0]
        data_4d = np.reshape(getattr(self, data_attr), (-1, ) + tuple(img_shape))

        # add padding if necessary
        pad_width = calc_padding_size(img_shape, patch_shape, stride)
        if pad_width is not None:
            data_padded = np.pad(data_4d, pad_width, mode=padding)
        else:
            data_padded = data_4d

        # init tensor with patches
        num_sections = (np.asarray(data_padded.shape[1:]) - patch_shape) // stride + 1
        patches = np.zeros(shape=(len(self), np.prod(num_sections)) + tuple(patch_shape))

        # put patches into the tensor
        fake = np.zeros(len(self))
        get_patches_numba(data_padded, patch_shape, stride, patches, fake)
        patches = np.reshape(patches, (len(self) * np.prod(num_sections), ) + tuple(patch_shape))
        return patches

    def load_from_patches(self, patches, stride, scan_shape, data_attr='images'):
        """ Assemble skyscraper from 4d-array of patches and
                put it into data_attr-attribute of batch

        Args:
            patches: 4d-array of patches, first dim enumerates patches
                others are spatial in order (z, y, x)
            scan_shape: tuple/ndarray/list of len=3 with shape of scan-array
                patches are assembled into; order of axis is (z, y, x)
            stride: tuple/ndarray/list of len=3 with stride with which
                patches are put into data_attr;
                if stride != patch shape, averaging over overlapping regions
                    is used
            data_attr: the name of attribute, the assembled skyscraper should
                be put into
            *NOTE: scan_shape, patches.shape, sride are used to infer the number
                of sections;
                in case of overshoot we crop the padding out

        Return:
            ___
        """
        scan_shape, stride = np.asarray(scan_shape), np.asarray(stride)
        patch_shape = np.asarray(patches.shape[1:])

        # infer what padding was applied to scans when extracting patches
        pad_width = calc_padding_size(scan_shape, patch_shape, stride)

        # if padding is non-zero, adjust the shape of scan
        if pad_width is not None:
            shape_delta = np.asarray([before + after for before, after in pad_width[1:]])
        else:
            shape_delta = np.zeros(3).astype('int')

        scan_shape_adj = scan_shape + shape_delta

        # init 4d tensor and put assembled scans into it
        data_4d = np.zeros((len(self), ) + tuple(scan_shape_adj))
        patches = np.reshape(patches, (len(self), -1) + tuple(patch_shape))
        fake = np.zeros(len(self))
        assemble_patches(patches, stride, data_4d, fake)

        # crop (perform anti-padding) if necessary
        if pad_width is not None:
            data_shape = data_4d.shape
            slc_z = slice(pad_width[1][0], data_shape[1] - pad_width[1][1])
            slc_y = slice(pad_width[2][0], data_shape[2] - pad_width[2][1])
            slc_x = slice(pad_width[3][0], data_shape[3] - pad_width[3][1])
            data_4d = data_4d[:, slc_z, slc_y, slc_x]

        # reshape 4d-data to skyscraper form and put it into needed attr
        data_4d = data_4d.reshape((len(self) * scan_shape[0], ) + tuple(scan_shape[1:]))
        setattr(self, data_attr, data_4d)

    @action
    def normalize_hu(self, min_hu=-1000, max_hu=400):
        """ Normalize hu-densities to interval [0, 255]:
                trim hus outside range [min_hu, max_hu], then scale to [0, 255].

        Args:


        Example:
            batch = batch.normalize_hu(min_hu=-1300, max_hu=600)
        """
        # trimming and scaling to [0, 1]
        self.images = (self.images - min_hu) / (max_hu - min_hu)
        self.images[self.images > 1] = 1.
        self.images[self.images < 0] = 0.

        # scaling to [0, 255]
        self.images *= 255
        return self

    @action
    def double_normalize_hu(self, min_hu=-1000, max_hu=400):
        """ Call normalize_hu action twice.

        This method is required by KerasUnet model.
        """
        self.normalize_hu(min_hu, max_hu)
        self.normalize_hu(min_hu, max_hu)
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
        """ Get axial slice (e.g., for plots)

        Args:
            person_number - can be either number of person in the batch
                or index of the person whose axial slice we need
            slice_height: e.g. 0.7 means that we take slice with number
                int(0.7 * number of slices for person)

        example: patch = batch.get_axial_slice(5, 0.6)
                 patch = batch.get_axial_slice(self.index[5], 0.6)
                 # here self.index[5] usually smth like 'a1de03fz29kf6h2'

        """
        margin = int(slice_height * self.get(person_number, 'images').shape[0])
        patch = self.get(person_number, 'images')[margin, :, :]
        return patch
