# pylint: disable=too-many-arguments
# pylint: disable=undefined-variable
# pylint: disable=no-member

""" Batch class for storing CT-scans. """

import os
import logging
import dill as pickle

import numpy as np
import aiofiles
import blosc
import dicom
import SimpleITK as sitk

from ..dataset import Batch, action, inbatch_parallel, any_action_failed, DatasetIndex  # pylint: disable=no-name-in-module

from .resize import resize_scipy, resize_pil
from .segment import calc_lung_mask_numba
from .mip import make_xip_numba, numba_xip
from .flip import flip_patient_numba
from .crop import make_central_crop
from .patches import get_patches_numba, assemble_patches, calc_padding_size
from .rotate import rotate_3D
from .dump import dump_data

# logger initialization
logger = logging.getLogger(__name__) # pylint: disable=invalid-name

AIR_HU = -2000
DARK_HU = -2000

class CTImagesBatch(Batch):  # pylint: disable=too-many-public-methods
    """ Batch class for storing batch of CT-scans in 3D.

    Contains a component `images` = 3d-array of stacked scans
    along number_of_slices (z) axis (aka "skyscraper"), associated information
    for subsetting individual patient's 3D scan (_bounds, origin, spacing) and
    various methods to preprocess the data.

    Parameters
    ----------
    index : dataset.index
        ids of scans to be put in a batch

    Attributes
    ----------
    components : tuple of strings.
        List names of data components of a batch, which are `images`,
        `origin` and `spacing`.
        NOTE: Implementation of this attribute is required by Base class.
    index : dataset.index
        represents indices of scans from a batch
    images : ndarray
        contains ct-scans for all patients in batch.
    spacing : ndarray of floats
        represents distances between pixels in world coordinates
    origin : ndarray of floats
        contains world coordinates of (0, 0, 0)-pixel of scans
    """

    components = "images", "spacing", "origin"

    def __init__(self, index, *args, **kwargs):
        """ Execute Batch construction and init of basic attributes

        Parameters
        ----------
        index : Dataset.Index class.
            Required indexing of objects (files).
        """

        super().__init__(index, *args, **kwargs)

        # init basic attrs
        self.images = None
        self._bounds = None
        self.origin = None
        self.spacing = None
        self._init_data(spacing=np.ones(shape=(len(self), 3)),
                        origin=np.zeros(shape=(len(self), 3)),
                        bounds=np.array([], dtype='int'))

    def _if_component_filled(self, component):
        """ Check if component is filled with data.

        Parameters
        ----------
        component : str
            component to be checked

        Returns
        -------
        bool
            True if filled, False if not.
        """
        return getattr(self, component, None) is not None

    def _init_data(self, bounds=None, **kwargs):
        """ Initialize _bounds and components (images, origin, spacing).

        `_init_data` is used as initializer of batch inner structures,
        called inside __init__ and other methods

        Parameters
        ----------
        **kwargs
            images : ndarray(n_patients * z, y, x) or None
                data to be put as a component `images` in self.images, where
                n_patients is total number of patients in array and `z, y, x`
                is a shape of each patient 3D array.
                Note, that each patient should have same and constant
                `z, y, x` shape.
            bounds : ndarray(n_patients, dtype=np.int) or None
                1d-array of bound-floors for each scan 3D array,
                has length = number of items in batch + 1, to be put in self._bounds.
            origin : ndarray(n_patients, 3) or None
                2d-array contains origin coordinates of patients scans
                in `z,y,x`-format in world coordinates to be put in self.origin.
            spacing : ndarray(n_patients, 3) or None
                2d-array [number of items X 3] of spacings between slices
                along each of `z,y,x` axes for each patient's 3D array
                in world coordinates to be put in self.spacing.
        """
        self._bounds = bounds if bounds is not None else self._bounds
        for comp_name, comp_data in kwargs.items():
            setattr(self, comp_name, comp_data)

    @classmethod
    def split(cls, batch, batch_size):
        """ Split one batch in two batches.

        The lens of 2 batches would be `batch_size` and `len(batch) - batch_size`

        Parameters
        ----------
        batch : Batch class instance
            batch to be splitted in two
        batch_size : int
            length of first returned batch.
            If batch_size >= len(batch), return None instead of a 2nd batch

        Returns
        -------
        tuple of batches
            (1st_Batch, 2nd_Batch)


        Notes
        -----
        Method does not change the structure of input Batch.index. Indices of output
        batches are simply subsets of input Batch.index.
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
        """ Concatenate several batches in one large batch.

        Assume that same components are filled in all supplied batches.

        Parameters
        ----------
        batches : list or tuple of batches
            sequence of batches to be concatenated

        Returns
        -------
        batch
            large batch with length = sum of lengths of concated batches

        Notes
        -----
        Old batches' indexes are dropped. New large batch has new
        np-arange index.
        if None-entries or batches of len=0 are included in the list of batches,
        they will be dropped after concat.
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

        Parameters
        ----------
        batches : list of batches
        batch_size : int
            length of first resulting batch

        Returns
        -------
        tuple of batches
            (new_batch, rest_batch)

        Notes
        -----
        Merge performs split (of middle-batch) and then two concats
        because of speed considerations.
        """
        batches = [batch for batch in batches if batch is not None]
        batches = [batch for batch in batches if len(batch) > 0]
        if batch_size is None:
            return (cls.concat(batches), None)
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
    def load(self, fmt='dicom', components=None, bounds=None, **kwargs):      # pylint: disable=arguments-differ
        """ Load 3d scans data in batch.

        Parameters
        ----------
        fmt : str
            type of data. Can be 'dicom'|'blosc'|'raw'|'ndarray'
        components : tuple, list, ndarray of strings or str
            Contains names of batch component(s) that should be loaded.
            As of now, works only if fmt='blosc'. If fmt != 'blosc', all
            available components are loaded. If None and fmt = 'blosc', again,
            all components are loaded.
        bounds : ndarray(n_patients + 1, dtype=np.int) or None
            Needed iff fmt='ndarray'. Bound-floors for items from a `skyscraper`
            (stacked scans).
        **kwargs
            images : ndarray(n_patients * z, y, x) or None
                Needed only if fmt = 'ndarray'
                input array containing `skyscraper` (stacked scans).
            origin : ndarray(n_patients, 3) or None
                Needed only if fmt='ndarray'.
                origins of scans in world coordinates.
            spacing : ndarray(n_patients, 3) or None
                Needed only if fmt='ndarray'
                ndarray with spacings of patients along `z,y,x` axes.

        Returns
        -------
        self

        Examples
        --------
        DICOM example
        initialize batch for storing batch of 3 patients with following IDs:

        >>> index = FilesIndex(path="/some/path/*.dcm", no_ext=True)
        >>> batch = CTImagesBatch(index)
        >>> batch.load(fmt='dicom')

        Ndarray example

        images_array stores a set of 3d-scans concatted along 0-zxis, "skyscraper".
        Say, it is a ndarray with shape (400, 256, 256)

        bounds stores ndarray of last floors for each scan.
        say, bounds = np.asarray([0, 100, 400])

        >>> batch.load(fmt='ndarray', images=images_array, bounds=bounds)

        """
        # if ndarray
        if fmt == 'ndarray':
            self._init_data(bounds=bounds, **kwargs)
        elif fmt == 'dicom':
            self._load_dicom()              # pylint: disable=no-value-for-parameter
        elif fmt == 'blosc':
            components = self.components if components is None else components
            # convert components_blosc to iterable
            components = np.asarray(components).reshape(-1)

            self._load_blosc(components=components)              # pylint: disable=no-value-for-parameter
        elif fmt == 'raw':
            self._load_raw()                # pylint: disable=no-value-for-parameter
        else:
            raise TypeError("Incorrect type of batch source")
        return self

    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_dicom(self, patient_id, **kwargs):
        """ Read dicom file, load 3d-array and convert to Hounsfield Units (HU).

        Notes
        -----
        Conversion to hounsfield unit scale using meta from dicom-scans is performed.
        """
        # put 2d-scans for each patient in a list
        patient_pos = self.index.get_pos(patient_id)
        patient_folder = self.index.get_fullpath(patient_id)
        list_of_dicoms = [dicom.read_file(os.path.join(patient_folder, s))
                          for s in os.listdir(patient_folder)]

        list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)

        dicom_slice = list_of_dicoms[0]
        intercept_pat = dicom_slice.RescaleIntercept
        slope_pat = dicom_slice.RescaleSlope

        self.spacing[patient_pos, ...] = np.asarray([float(dicom_slice.SliceThickness),
                                                     float(dicom_slice.PixelSpacing[0]),
                                                     float(dicom_slice.PixelSpacing[1])], dtype=np.float)

        self.origin[patient_pos, ...] = np.asarray([float(dicom_slice.ImagePositionPatient[2]),
                                                    float(dicom_slice.ImagePositionPatient[0]),
                                                    float(dicom_slice.ImagePositionPatient[1])], dtype=np.float)

        patient_data = np.stack([s.pixel_array for s in list_of_dicoms]).astype(np.int16)

        patient_data[patient_data == AIR_HU] = 0

        # perform conversion to HU
        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)
        return patient_data

    def _prealloc_skyscraper_components(self, components, fmt='blosc'):
        """ Read shapes of skyscraper-components dumped with blosc,
        allocate memory for them, update self._bounds.

        Used for more efficient load in terms of memory.

        Parameters
        ---------
        components : str or iterable
            iterable of components we need to preload.
        fmt : str
            format in which components are stored on disk.

        """
        if fmt != 'blosc':
            raise NotImplementedError('Preload from {} not implemented yet'.format(fmt))

        # make iterable out of components-arg
        components = [components] if isinstance(components, str) else list(components)

        # load shapes, perform memory allocation
        for component in components:
            shapes = np.zeros((len(self), 3), dtype=np.int)
            for ix in self.indices:
                filename = os.path.join(self.index.get_fullpath(ix), component, 'data.shape')
                ix_pos = self._get_verified_pos(ix)

                # read shape and put it into shapes
                if not os.path.exists(filename):
                    raise OSError("Component {} for item {} cannot be found on disk".format(component, ix))
                with open(filename, 'rb') as file:
                    shapes[ix_pos, :] = pickle.load(file)

            # update bounds of items
            # TODO: once bounds for other components are added, make sure they are updated here in the right way
            self._bounds = np.cumsum(np.insert(shapes[:, 0], 0, 0), dtype=np.int)

            # preallocate the component
            skysc_shape = (self._bounds[-1], shapes[0, 1], shapes[0, 2])
            setattr(self, component, np.zeros(skysc_shape))

    def _init_load_blosc(self, **kwargs):
        """ Init-function for load from blosc.

        Parameters
        ----------
        **kwargs
            components : iterable of components that need to be loaded

        Returns
        -------
        list
            list of ids of batch-items
        """
        # set images-component to 3d-array of zeroes if the component is to be updated
        if 'images' in kwargs['components']:
            self._prealloc_skyscraper_components('images')

        return self.indices

    @inbatch_parallel(init='_init_load_blosc', post='_post_default', target='async', update=False)
    async def _load_blosc(self, ix, *args, **kwargs):
        """ Read scans from blosc and put them into batch components

        Parameters
        ----------
        **kwargs
            components : tuple
                tuple of strings with names of components of data
                that should be loaded into self

        Notes
        -----
        NO conversion to HU is done for blosc
        (because usually it's done before).
        """

        for source in kwargs['components']:
            # set correct extension for each component and choose a tool
            # for debyting and (possibly) decoding it
            if source in ['spacing', 'origin']:
                ext = 'pkl'
                unpacker = pickle.loads
            else:
                ext = 'blk'
                def unpacker(byted):
                    """ Debyte and decode an ndarray
                    """
                    debyted = blosc.unpack_array(byted)

                    # read the decoder and apply it
                    decod_path = os.path.join(self.index.get_fullpath(ix), source, 'data.decoder')

                    # if file with decoder not exists, assume that no decoding is needed
                    if os.path.exists(decod_path):
                        with open(decod_path, mode='rb') as file:
                            decoder = pickle.loads(file.read())
                    else:
                        decoder = lambda x: x

                    return decoder(debyted)

            comp_path = os.path.join(self.index.get_fullpath(ix), source, 'data' + '.' + ext)
            if not os.path.exists(comp_path):
                raise OSError("File with component {} doesn't exist".format(source))

            # read the component
            async with aiofiles.open(comp_path, mode='rb') as file:
                byted = await file.read()

            # de-byte it with the chosen tool
            component = unpacker(byted)

            # update needed slice(s) of component
            comp_pos = self.get_pos(None, source, ix)
            getattr(self, source)[comp_pos] = component

        return None

    def _load_raw(self, **kwargs):        # pylint: disable=unused-argument
        """ Load scans from .raw images (with meta in .mhd)

        Notes
        -----
        Method does NO conversion to HU
        NO multithreading is used, as SimpleITK (sitk) lib crashes
        in multithreading mode in experiments.
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

    @action
    @inbatch_parallel(init='_init_dump', post='_post_default', target='async', update=False)
    async def dump(self, ix, dst, components=None, fmt='blosc', index_to_name=None, i8_encoding_mode=None):
        """ Dump chosen ``components`` of scans' batcn in folder ``dst`` in specified format.

        When some of the ``components`` are ``None``, a warning is printed and nothing is dumped.
        By default (``components is None``) ``dump`` attempts to dump all components.

        Parameters
        ----------
        dst : str
            destination-folder where all patients' data should be put
        components : tuple, list, ndarray of strings or str
            component(s) that we need to dump (smth iterable or string). If not
            supplied, dump all components
        fmt : 'blosc'
            format of dump. Currently only blosc-format is supported;
            in this case folder for each patient is created. Tree-structure of created
            files is demonstrated in the example below.
        index_to_name : callable or None
            When supplied, should return str;
            A function that relates each item's index to a name of item's folder.
            That is, each item is dumped into os.path.join(dst, index_to_name(items_index)).
            If None, no transformation is applied and the method attempts to use indices of batch-items
            as names of items' folders.
        i8_encoding_mode : int, str or dict
            whether (and how) components of skyscraper-type should be cast to int8.
            If None, no cast is performed. The cast allows to save space on disk and to speed up batch-loading.
            However, it comes with loss of precision, as originally skyscraper-components are stored
            in float32-format. Can be int: 0, 1, 2 or str/None: 'linear', 'quantization' or None.
            0 or None disable the cast. 1 stands for 'linear', 2 - for 'quantization'.
            Can also be component-wise dict of modes, e.g.: {'images': 'linear', 'masks': 0}.

        Examples
        --------
        Initialize batch and load data

        >>> ind = ['1ae34g90', '3hf82s76']
        >>> batch = CTImagesBatch(ind)
        >>> batch.load(...)
        >>> batch.dump(dst='./data/blosc_preprocessed')

        The command above creates following files:

        - ./data/blosc_preprocessed/1ae34g90/images/data.blk
        - ./data/blosc_preprocessed/1ae34g90/images/data.shape
        - ./data/blosc_preprocessed/1ae34g90/spacing/data.pkl
        - ./data/blosc_preprocessed/1ae34g90/origin/data.pkl

        - ./data/blosc_preprocessed/3hf82s76/images/data.blk
        - ./data/blosc_preprocessed/3hf82s76/images/data.shape
        - ./data/blosc_preprocessed/3hf82s76/spacing/data.pkl
        - ./data/blosc_preprocessed/3hf82s76/origin/data.pkl
        """
        # if components-arg is not supplied, dump all components
        if components is None:
            components = self.components

        if fmt != 'blosc':
            raise NotImplementedError('Dump to {} is not implemented yet'.format(fmt))

        # make sure that components is iterable
        components = np.asarray(components).reshape(-1)
        data_items = dict()

        for component in components:
            # get correct extension for the component
            if component in ['spacing', 'origin']:
                ext = 'pkl'
            else:
                ext = 'blk'

            # get component belonging to the needed item, add it to items-dict
            comp_pos = self.get_pos(None, component, ix)
            data_items.update({component: [getattr(self, component)[comp_pos], ext]})

        # set item-specific folder
        item_subdir = ix if index_to_name is None else index_to_name(ix)
        item_dir = os.path.join(dst, item_subdir)

        return await dump_data(data_items, item_dir, i8_encoding_mode)

    def get_pos(self, data, component, index):
        """ Return a positon of an item for a given index in data
        or in self.`component`.

        Fetch correct position inside batch for an item, looks for it
        in `data`, if provided, or in `component` in self.

        Parameters
        ----------
        data : None or ndarray
            data from which subsetting is done.
            If None, retrieve position from `component` of batch,
            if ndarray, returns index.
        component : str
            name of a component, f.ex. 'images'.
            if component provided, data should be None.
        index : str or int
            index of an item to be looked for.
            may be key from dataset (str)
            or index inside batch (int).

        Returns
        -------
        int
            Position of item

        Notes
        -----
        This is an overload of get_pos from base Batch-class,
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
        """ Get position of patient in batch.

        Whatever index is passed in this method, it returns
        corresponding index inside batch.

        Parameters
        ----------
        index : str or int
            Can be either position of patient in self.images
            or index from self.index. If int, it means that
            index is already patient's position in Batch.
            If str, it's handled as a key, and returns a position in batch.
            If fetched position is out of bounds then Exception is generated.

        Returns
        -------
        int
            patient's position inside batch
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
        """ Get shapes for all 3d scans in CTImagesBatch.

        Returns
        -------
        ndarray
            shapes of data for each patient, ndarray(patient_pos, 3)
        """
        shapes = np.zeros((len(self), 3), dtype=np.int)
        shapes[:, 0] = self.upper_bounds - self.lower_bounds
        shapes[:, 1], shapes[:, 2] = self.slice_shape
        return shapes

    @property
    def lower_bounds(self):
        """ Get lower bounds of patients data in CTImagesBatch.

        Returns
        -------
        ndarray
            ndarray(n_patients,) containing
            lower bounds of patients data along z-axis.
        """
        return self._bounds[:-1]

    @property
    def upper_bounds(self):
        """ Get upper bounds of patients data in CTImagesBatch.

        Returns
        -------
        ndarray
            ndarray(n_patients,) containing
            upper bounds of patients data along z-axis.
        """
        return self._bounds[1:]

    @property
    def slice_shape(self):
        """ Get shape of slice in yx-plane.

        Returns
        -------
        ndarray
            ndarray([y_dim, x_dim],dtype=np.int) with shape of scan slice.
        """
        return np.asarray(self.images.shape[1:])

    def rescale(self, new_shape):
        """ Recomputes spacing values for patients' data after resize.

        Parameters
        ----------
        new_shape : ndarray(dtype=np.int)
            shape of patient 3d array after resize,
            in format np.array([z_dim, y_dim, x_dim], dtype=np.int).

        Returns
        -------
        ndarray
            ndarray(n_patients, 3) with spacing values for each
            patient along z, y, x axes.
        """
        return (self.spacing * self.images_shape) / new_shape

    def _reraise_worker_exceptions(self, worker_outputs):
        """ Reraise exceptions coming from worker-functions, if there are any.

        Parameters
        ----------
        worker_outputs : list
            list of workers' results
        """
        if any_action_failed(worker_outputs):
            all_errors = self.get_errors(worker_outputs)
            raise RuntimeError("Failed parallelizing. Some of the workers failed with following errors: ", all_errors)

    def _post_default(self, list_of_arrs, update=True, new_batch=False, **kwargs):
        """ Gatherer outputs of different workers, update `images` component

        Parameters
        ----------
        list_of_arrs : list
            list of ndarrays to be concated and put in a batch.images.
        update : bool
            if False, nothing is performed.
        new_batch : bool
            if False, empty batch is created,
            if True, data is gathered, loaded and put into batch.images.

        Returns
        -------
        batch
            new batch, empty batch or self-batch.

        Notes
        -----
        Output of each worker should correspond to individual patient.
        """
        self._reraise_worker_exceptions(list_of_arrs)
        res = self
        if update:
            new_data = np.concatenate(list_of_arrs, axis=0)
            new_bounds = np.cumsum(np.array([len(a) for a in [[]] + list_of_arrs]))
            params = dict(images=new_data, bounds=new_bounds,
                          origin=self.origin, spacing=self.spacing)
            if new_batch:
                batch = type(self)(self.index)
                batch.load(fmt='ndarray', **params)
                res = batch
            else:
                self._init_data(**params)
        return res

    def _post_components(self, list_of_dicts, **kwargs):
        """ Gather outputs of different workers, update many components.

        Parameters
        ----------
        list_of_dicts : list
            list of dicts {`component_name`: what_to_place_in_component}

        Returns
        -------
        self
            changes self's components
        """
        self._reraise_worker_exceptions(list_of_dicts)

        # if images is in dict, update bounds
        if 'images' in list_of_dicts[0]:
            list_of_images = [worker_res['images'] for worker_res in list_of_dicts]
            new_bounds = np.cumsum(np.array([len(a) for a in [[]] + list_of_images]))
            new_data = np.concatenate(list_of_images, axis=0)
            params = dict(images=new_data, bounds=new_bounds,
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
        """ Fetch args for loading `images` using inbatch_parallel.

        Args-fetcher for parallelization using inbatch_parallel.

        Returns
        -------
        list
            list of patient's 3d arrays.
        """
        return [self.get(patient_id, 'images') for patient_id in self.indices]

    def _init_rebuild(self, **kwargs):
        """ Fetch args for `images` rebuilding using inbatch_parallel.


        Args-fetcher for parallelization using inbatch_parallel

        Parameters
        ----------
        **kwargs
                shape : tuple, list or ndarray of int
                    (z,y,x)-shape of every image in image component after action is performed.
                spacing : tuple, list or ndarray of float
                    (z,y,x)-spacing for each image. If supplied, assume that
                    unify_spacing is performed.

        Returns
        -------
        list
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
                item_args['factor'] = self.spacing[i, :] / np.array(kwargs['spacing'])
                item_args['shape_resize'] = shape_after_resize[i, :]

            all_args += [item_args]

        return all_args

    def _init_dump(self, **kwargs):
        """ Init function for dump.

        Checks if all components that should be dumped are non-None. If some are None,
        prints warning and makes sure that nothing is dumped.

        Parameters
        ----------
        **kwargs:
            components : tuple, list, ndarray of strings or str
                components that we need to dump
        """
        components = kwargs.get('components', self.components)

        # make sure that components is iterable
        components = np.asarray(components).reshape(-1)

        _empty = [component for component in components if not self._if_component_filled(component)]

        # if some of the components for dump are empty, print warning and do not dump anything
        if len(_empty) > 0:
            logger.warning('Components %r are empty. Nothing is dumped!', _empty)
            return []
        else:
            return self.indices

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):
        """ Gather outputs of different workers for actions, rebuild `images` component.

        Parameters
        ----------
        all_outputs : list
            list of outputs. Each item is given by tuple
        new_batch : bool
            if True, returns new batch with data agregated
            from all_ouputs. if False, changes self.
        **kwargs
                shape : list, tuple or ndarray of int
                    (z,y,x)-shape of every image in image component after action is performed.
                spacing : tuple, list or ndarray of float
                    (z,y,x)-spacing for each image. If supplied, assume that
                    unify_spacing is performed.
        """
        self._reraise_worker_exceptions(all_outputs)

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
        params = dict(images=new_data, bounds=new_bounds,
                      origin=new_origin, spacing=new_spacing)
        if new_batch:
            batch_res = type(self)(self.index)
            batch_res.load(fmt='ndarray', **params)
            return batch_res
        else:
            self._init_data(**params)
            return self

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='threads')
    def resize(self, patient, out_patient, res, shape=(128, 256, 256), method='pil-simd',
               axes_pairs=None, resample=None, order=3, *args, **kwargs):
        """ Resize (change shape of) each CT-scan in the batch.

        When called from a batch, changes this batch.

        Parameters
        ----------
        shape : tuple, list or ndarray of int
            (z,y,x)-shape that should be AFTER resize.
            Note, that ct-scan dim_ordering also should be `z,y,x`
        method : str
            interpolation package to be used. Either 'pil-simd' or 'scipy'.
            Pil-simd ensures better quality and speed on configurations
            with average number of cores. On the contrary, scipy is better scaled and
            can show better performance on systems with large number of cores
        axes_pairs : None or list/tuple of tuples with pairs
            pairs of axes that will be used for performing pil-simd resize,
            as this resize is made in 2d. Min number of pairs to use is 1,
            at max there can be 6 pairs. If None, set to ((0, 1), (1, 2)).
            The more pairs one uses, the more precise is the result.
            (and computation takes more time).
        resample : filter of pil-simd resize. By default set to bilinear. Can be any of filters
            supported by PIL.Image.
        order : the order of scipy-interpolation (<= 5)
            large value improves precision, but slows down the computaion.

        Examples
        --------
        >>> shape = (128, 256, 256)
        >>> batch = batch.resize(shape=shape, order=2, method='scipy')
        >>> batch = batch.resize(shape=shape, resample=PIL.Image.BILINEAR)
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
    def unify_spacing(self, patient, out_patient, res, factor,
                      shape_resize, spacing=(1, 1, 1), shape=(128, 256, 256),
                      method='pil-simd', order=3, padding='edge', axes_pairs=None,
                      resample=None, *args, **kwargs):
        """ Unify spacing of all patients.

        Resize all patients to meet `spacing`, then crop/pad resized array to meet `shape`.

        Parameters
        ----------
        spacing : tuple, list or ndarray of float
            (z,y,x)-spacing after resize.
            Should be passed as key-argument.
        shape : tuple, list or ndarray of int
            (z,y,x)-shape after crop/pad.
            Should be passed as key-argument.
        method : str
            interpolation method ('pil-simd' or 'resize').
            Should be passed as key-argument.
            See CTImagesBatch.resize for more information.
        order : None or int
            order of scipy-interpolation (<=5), if used.
            Should be passed as key-argument.
        padding : str
            mode of padding, any supported by np.pad.
            Should be passed as key-argument.
        axes_pairs : tuple, list of tuples with pairs
            pairs of axes that will be used consequentially
            for performing pil-simd resize.
            Should be passed as key-argument.
        resample : None or str
            filter of pil-simd resize.
            Should be passed as key-argument
        patient : str
            index of patient, that worker is handling.
            Note: this argument is passed by inbatch_parallel
        out_patient : ndarray
            result of individual worker after action.
            Note: this argument is passed by inbatch_parallel
        res : ndarray
            New `images` to replace data inside `images` component.
            Note: this argument is passed by inbatch_parallel
        factor : tuple
            (float), factor to make resize by.
            Note: this argument is passed by inbatch_parallel
        shape_resize : tuple
            It is possible to provide `shape_resize` argument (shape after resize)
            instead of spacing. Then array with `shape_resize`
            will be cropped/padded for shape to = `shape` arg.
            Note that this argument is passed by inbatch_parallel

        Notes
        -----
        see CTImagesBatch.resize for more info about methods' params.

        Examples
        --------
        >>> shape = (128, 256, 256)
        >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                        order=2, method='scipy', padding='reflect')
        >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                        resample=PIL.Image.BILINEAR)
        """
        if method == 'scipy':
            args_resize = dict(patient=patient, out_patient=out_patient,
                               res=res, order=order, factor=factor, padding=padding)
            return resize_scipy(**args_resize)
        elif method == 'pil-simd':
            args_resize = dict(input_array=patient, output_array=out_patient,
                               res=res, axes_pairs=axes_pairs, resample=resample,
                               shape_resize=shape_resize, padding=padding)
            return resize_pil(**args_resize)

    @action
    @inbatch_parallel(init='indices', post='_post_default', update=False, target='threads')
    def rotate(self, index, angle, components='images', axes=(1, 2), random=True, **kwargs):
        """ Rotate 3D images in batch on specific angle in plane.

        Parameters
        ----------
        angle : float
            degree of rotation.
        components : tuple, list, ndarray of strings or str
            name(s) of components to rotate each item in it.
        axes : tuple, list or ndarray of int
            (int, int), plane of rotation specified by two axes (zyx-ordering).
        random : bool
            if True, then degree specifies maximum angle of rotation.

        Returns
        -------
        ndarray
            ndarray of 3D rotated image.

        Notes
        -----
        zero padding automatically added after rotation.
        Use this action in the end of pipelines for purposes of augmentation.
        E.g., after :func:`~radio.preprocessing.ct_masked_batch.CTImagesMaskedBatch.sample_nodules`

        Examples
        --------
        Rotate images on 90 degrees:

        >>> batch = batch.rotate(angle=90, axes=(1, 2), random=False)

        Random rotation with maximum angle:

        >>> batch = batch.rotate(angle=30, axes=(1, 2))

        """
        _components = np.asarray(components).reshape(-1)
        _angle = angle * np.random.rand() if random else angle
        for comp in _components:
            data = self.get(index, comp)
            rotate_3D(data, _angle, axes)

    @inbatch_parallel(init='_init_images', post='_post_default', target='threads', new_batch=True)
    def _make_xip(self, image, depth, stride=2, mode='max',
                  projection='axial', padding='reflect', *args, **kwargs):
        """ Make intensity projection (maximum, minimum, mean or median).

        Notice that axis is chosen according to projection argument.

        Parameters
        ----------
        depth : int
            number of slices over which xip operation is performed.
        stride : int
            stride-step along projection dimension.
        mode : str
            Possible values are 'max', 'min', 'mean' or 'median'.
        projection : str
            Possible values: 'axial', 'coronal', 'sagital'.
            In case of 'coronal' and 'sagital' projections tensor
            will be transposed from [z,y,x] to [x,z,y] and [y,z,x].
        padding : str
            mode of padding that will be passed in numpy.padding function.
        """
        return make_xip_numba(image, depth, stride, mode, projection, padding)

    @action
    def make_xip(self, depth, stride=1, mode='max', projection='axial', padding='reflect', **kwargs):
        """ Make intensity projection (maximum, minimum, mean or median).

        Notice that axis is chosen according to projection argument.

        Parameters
        ----------
        depth : int
            number of slices over which xip operation is performed.
        stride : int
            stride-step along projection dimension.
        mode : str
            Possible values are 'max', 'min', 'mean' or 'median'.
        projection : str
            Possible values: 'axial', 'coronal', 'sagital'.
            In case of 'coronal' and 'sagital' projections tensor
            will be transposed from [z,y,x] to [x,z,y] and [y,z,x].
        padding : str
            mode of padding that will be passed in numpy.padding function.
        """
        output_batch = self._make_xip(depth=depth, stride=stride, mode=mode,  # pylint: disable=no-value-for-parameter
                                      projection=projection, padding=padding)
        output_batch.spacing = self.rescale(output_batch.images_shape)
        return output_batch

    def xip_component(self, component, mode, depth, stride, start=0, channels=None, squeeze=False):
        """ Make channelled intensity projection from a component.
        """
        # parse arguments
        _modes = {'max': 0, 'min': 1, 'mean': 2, 'median': 3}
        mode = mode if isinstance(mode, (list, tuple)) else (mode, )
        mode = [m if isinstance(m, int) else _modes[m] for m in mode]
        channels = 1 if channels is None else channels

        # loop over batch-items and modes
        items = []
        for ix in self.indices:
            image = self.get(ix, component)
            xips = []
            for m in mode:
                xip = numba_xip(image, depth, m, stride, start)
                if channels == 1:
                    xip = np.expand_dims(xip, axis=-1)
                else:
                    # split xip into channels
                    rem = len(image) % channels
                    if rem > 0:
                        xip = xip[:-rem]

                    xip = xip.reshape((-1, channels) + xip.shape[1:])
                    xip = np.transpose(xip, (0, 2, 3, 1))
                    xip = np.max(xip, axis=-1, keepdims=True) if squeeze else xip
                xips.append(xip)

            item = np.concatenate(xips, axis=-1)
            items.append(item)

        return np.concatenate(items, axis=0)


    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='threads', new_batch=True)
    def calc_lung_mask(self, patient, out_patient, res, erosion_radius, **kwargs):     # pylint: disable=unused-argument, no-self-use
        """ Return a mask for lungs

        Parameters
        ----------
        erosion_radius : int
            radius of erosion to be performed.
        """
        return calc_lung_mask_numba(patient, out_patient, res, erosion_radius)

    @action
    def segment(self, erosion_radius=2, **kwargs):
        """ Segment lungs' content from 3D array.

        Parameters
        ---------
        erosion_radius : int
            radius of erosion to be performed.

        Returns
        -------
        batch

        Notes
        -----
        Sets HU of every pixel outside lungs to DARK_HU = -2000.

        Examples
        --------

        >>> batch = batch.segment(erosion_radius=4, num_threads=20)
        """
        # get mask with specified params, apply it to scans
        mask_batch = self.calc_lung_mask(erosion_radius=erosion_radius, **kwargs)  # pylint: disable=no-value-for-parameter
        lungs_mask = mask_batch.images
        self.images *= lungs_mask

        # reverse the mask and set not-lungs to DARK_HU
        result_mask = 1 - lungs_mask
        result_mask *= DARK_HU

        self.images += result_mask

        return self

    @action
    def central_crop(self, crop_size, **kwargs):
        """ Make crop of crop_size from center of images.

        Parameters
        ----------
        crop_size : tuple, list or ndarray of int
            (z,y,x)-shape of crop.

        Returns
        -------
        batch
        """
        crop_size = np.asarray(crop_size).reshape(-1)
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
        """ Extract patches of patch_shape with specified stride.

        Parameters
        ----------
        patch_shape : tuple, list or ndarray of int
            (z,y,x)-shape of a single patch.
        stride : tuple, list or ndarray of int
            (z,y,x)-stride to slide over each patient's data.
        padding : str
            padding-type (see doc of np.pad for available types).
        data_attr : str
            component to get data from.

        Returns
        -------
        ndarray
            4d-ndaray of patches; first dimension enumerates patches

        Notes
        -----
        Shape of all patients data is needed to be the same at this step,
        resize/unify_spacing is required before.
        """

        patch_shape = np.asarray(patch_shape).reshape(-1)
        stride = np.asarray(stride).reshape(-1)

        img_shape = self.images_shape[0]
        data_4d = np.reshape(getattr(self, data_attr), (-1, *img_shape))

        # add padding if necessary
        pad_width = calc_padding_size(img_shape, patch_shape, stride)
        if pad_width is not None:
            data_padded = np.pad(data_4d, pad_width, mode=padding)
        else:
            data_padded = data_4d

        # init tensor with patches
        num_sections = (np.asarray(data_padded.shape[1:]) - patch_shape) // stride + 1
        patches = np.zeros(shape=(len(self), np.prod(num_sections), *patch_shape))

        # put patches into the tensor
        fake = np.zeros(len(self))
        get_patches_numba(data_padded, patch_shape, stride, patches, fake)
        patches = np.reshape(patches, (len(self) * np.prod(num_sections), *patch_shape))
        return patches

    def load_from_patches(self, patches, stride, scan_shape, data_attr='images'):
        """ Get skyscraper from 4d-array of patches, put it to `data_attr` component in batch.

        Let reconstruct original skyscraper from patches (if same arguments are passed)

        Parameters
        ----------
        patches : ndarray
            4d-array of patches, with dims: `(num_patches, z, y, x)`.
        scan_shape : tuple, list or ndarray of int
            (z,y,x)-shape of individual scan (should be same for all scans).
        stride : tuple, list or ndarray of int
            (z,y,x)-stride step used for gathering data from patches.
        data_attr : str
            batch component name to store new data.

        Notes
        -----
        If stride != patch.shape(), averaging of overlapped regions is used.
        `scan_shape`, patches.shape(), `stride` are used to infer the number of items
        in new skyscraper. If patches were padded, padding is removed for skyscraper.

        """
        scan_shape = np.asarray(scan_shape).reshape(-1)
        stride = np.asarray(stride).reshape(-1)
        patch_shape = np.asarray(patches.shape[1:]).reshape(-1)

        # infer what padding was applied to scans when extracting patches
        pad_width = calc_padding_size(scan_shape, patch_shape, stride)

        # if padding is non-zero, adjust the shape of scan
        if pad_width is not None:
            shape_delta = np.asarray([before + after for before, after in pad_width[1:]])
        else:
            shape_delta = np.zeros(3).astype('int')

        scan_shape_adj = scan_shape + shape_delta

        # init 4d tensor and put assembled scans into it
        data_4d = np.zeros((len(self), *scan_shape_adj))
        patches = np.reshape(patches, (len(self), -1, *patch_shape))
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
        data_4d = data_4d.reshape((len(self) * scan_shape[0], *scan_shape[1:]))
        setattr(self, data_attr, data_4d)

    @action
    def normalize_hu(self, min_hu=-1000, max_hu=400):
        """ Normalize HU-densities to interval [0, 255].

        Trim HU that are outside range [min_hu, max_hu], then scale to [0, 255].

        Parameters
        ----------
        min_hu : int
            minimum value for hu that will be used as trimming threshold.
        max_hu : int
            maximum value for hu that will be used as trimming threshold.

        Returns
        -------
        batch

        Examples
        --------
        >>> batch = batch.normalize_hu(min_hu=-1300, max_hu=600)
        """
        # trimming and scaling to [0, 1]
        self.images = (self.images - min_hu) / (max_hu - min_hu)
        self.images[self.images > 1] = 1.
        self.images[self.images < 0] = 0.

        # scaling to [0, 255]
        self.images *= 255
        return self

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='threads')
    def flip(self, patient, out_patient, res):    # pylint: disable=no-self-use
        """ Invert the order of slices for each patient

        Returns
        -------
        batch

        Examples
        --------
        >>> batch = batch.flip()
        """
        return flip_patient_numba(patient, out_patient, res)

    def get_axial_slice(self, person_number, slice_height):
        """ Get axial slice (e.g., for plots)

        Parameters
        ----------
        person_number : str or int
            Can be either index (int) of person in the batch
            or patient_id (str)
        slice_height : float
            scaled from 0 to 1 number of slice.
            e.g. 0.7 means that we take slice with number
            int(0.7 * number of slices for person)

        Returns
        -------
        ndarray (view)

        Examples
        --------
        Here self.index[5] usually smth like 'a1de03fz29kf6h2'

        >>> patch = batch.get_axial_slice(5, 0.6)
        >>> patch = batch.get_axial_slice(self.index[5], 0.6)

        """
        margin = int(slice_height * self.get(person_number, 'images').shape[0])
        patch = self.get(person_number, 'images')[margin, :, :]
        return patch
