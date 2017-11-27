# pylint: disable=too-many-arguments
# pylint: disable=undefined-variable
# pylint: disable=no-member

""" Batch class for storing CT-scans. """

import os
import cloudpickle

import numpy as np
import aiofiles
import blosc
import dicom
import SimpleITK as sitk
from sklearn.cluster import MiniBatchKMeans

from ..dataset import Batch, action, inbatch_parallel, any_action_failed, DatasetIndex  # pylint: disable=no-name-in-module

from .resize import resize_scipy, resize_pil
from .segment import calc_lung_mask_numba
from .mip import xip_fn_numba
from .flip import flip_patient_numba
from .crop import make_central_crop
from .patches import get_patches_numba, assemble_patches, calc_padding_size
from .rotate import rotate_3D


AIR_HU = -2000
DARK_HU = -2000
KMEANS_MINIBATCH = 10000
KMEANS_ITERS = 5


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
                 NOTE: Implementation of this property is required by Base class.
    index :      dataset.index
                 represents indices of scans from a batch
    images :     ndarray
                 contains ct-scans for all patients in batch.
    spacing :    ndarray of floats
                 represents distances between pixels in world coordinates
    origin :     ndarray of floats
                 contains world coordinates of (0, 0, 0)-pixel of scans
    """

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
        self._init_data()

    @property
    def components(self):
        """ Components' property.

        See doc of Base batch in dataset for information.

        Returns
        -------
        (str, str, str)
                       names of components returned from __getitem__.
        """
        return 'images', 'spacing', 'origin'

    def _init_data(self, source=None, bounds=None, origin=None, spacing=None):
        """ Initialize images, _bounds, origin and spacing attributes.

        `_init_data` is used as initializer of batch inner structures,
        called inside __init__ and other methods

        Parameters
        ----------
        source :  ndarray(n_patients * z, y, x) or None
                  data to be put as a component `images` in self.images, where
                  n_patients is total number of patients in array and `z, y, x`
                  is a shape of each patient 3D array.
                  Note, that each patient should have same and constant
                  `z, y, x` shape.
        bounds :  ndarray(n_patients, dtype=np.int) or None
                  1d-array of bound-floors for each scan 3D array,
                  has length = number of items in batch + 1, to be put in self._bounds.
        origin :  ndarray(n_patients, 3) or None
                  2d-array contains origin coordinates of patients scans
                  in `z,y,x`-format in world coordinates to be put in self.origin.
                  None value will be converted to zero-array.
        spacing : ndarray(n_patients, 3) or None
                  2d-array [number of items X 3] of spacings between slices
                  along each of `z,y,x` axes for each patient's 3D array
                  in world coordinates to be put in self.spacing.
                  None value will be converted to ones-array.
        """
        self.images = source
        self._bounds = bounds if bounds is not None else np.array([], dtype='int')
        self.origin = origin if origin is not None else np.zeros((len(self), 3))
        self.spacing = spacing if spacing is not None else np.ones((len(self), 3))

    @classmethod
    def split(cls, batch, batch_size):
        """ Split one batch in two batches.

        The lens of 2 batches would be `batch_size` and `len(batch) - batch_size`

        Parameters
        ----------
        batch :      Batch class instance
                     batch to be splitted in two
        batch_size : int
                     length of first returned batch.
                     If batch_size >= len(batch), return None instead of a 2nd batch

        Returns
        -------
        (1st_Batch, 2nd_Batch)


        Note
        ----
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

        Note
        ----
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
        batches :    list of batches
        batch_size : int
                     length of first resulting batch

        Returns
        -------
        (new_batch, rest_batch)

        Note
        ----
        Merge performs split (of middle-batch) and then two concats
        because of speed considerations.
        """
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
    def load(self, fmt='dicom', source=None, bounds=None,  # pylint: disable=arguments-differ
             origin=None, spacing=None, src_blosc=None):
        """ Load 3d scans data in batch.

        Parameters
        ----------
        fmt :       str
                    type of data. Can be 'dicom'|'blosc'|'raw'|'ndarray'
        source :    ndarray(n_patients * z, y, x) or None
                    input array with `skyscraper` (stacked scans),
                    needed iff fmt = 'ndarray'.
        bounds :    ndarray(n_patients, dtype=np.int) or None
                    bound-floors index for patients.
                    Needed iff fmt='ndarray'
        origin :    ndarray(n_patients, 3) or None
                    origins of scans in world coordinates.
                    Needed only if fmt='ndarray'
        spacing :   ndarray(n_patients, 3) or None
                    ndarray with spacings of patients along `z,y,x` axes.
                    Needed only if fmt='ndarray'
        src_blosc : list/tuple/string
                    Contains names of batch component(s) that should be loaded from blosc file.
                    Needed only if fmt='blosc'. If None, all components are loaded.

        Returns
        -------
        self

        Example
        -------
        DICOM example
        initialize batch for storing batch of 3 patients with following IDs:

        >>> index = FilesIndex(path="/some/path/*.dcm", no_ext=True)
        >>> batch = CTImagesBatch(index)
        >>> batch.load(fmt='dicom')

        Ndarray example

        Source_array stores a batch (concatted 3d-scans, skyscraper)
        say, ndarray with shape (400, 256, 256)

        bounds stores ndarray of last floors for each patient
        say, source_ubounds = np.asarray([0, 100, 400])

        >>> batch.load(source=source_array, fmt='ndarray', bounds=bounds)

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
        """ Read dicom file, load 3d-array and convert to Hounsfield Units (HU).

        Parameters
        ----------
        patient_id : str
                     patient dicom file index from batch, to be loaded and

        Returns
        -------
        ndarray
                3d-scan as np.ndarray

        Note
        ----
        Conversion to hounsfield unit scale using meta from dicom-scans is performed.
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

    def _preload_skyscraper_components(self, components):
        """ Read shapes of skyscraper-components dumped with blosc,
        allocate memory for them, update self._bounds.

        Used for more efficient load of blosc in terms of memory.

        Parameters
        ---------
        components : iterable of components we need to preload.

        """
        for component in components:
            shapes = np.zeros((len(self), 3), dtype=np.int)
            for ix in self.indices:
                filename = os.path.join(self.index.get_fullpath(ix), component, 'data.shape')
                ix_pos = self._get_verified_pos(ix)

                # read shape and put it into shapes
                with open(filename, 'rb') as file:
                    shapes[ix_pos, :] = cloudpickle.load(file)

            # update bounds of items
            # TODO: add bounds for other components
            self._bounds = np.cumsum(np.insert(shapes[:, 0], 0, 0), dtype=np.int)

            # fill the component with zeroes (memory preallocation)
            skysc_shape = (self._bounds[-1], shapes[0, 1], shapes[0, 2])
            setattr(self, component, np.zeros(skysc_shape))

    def _init_load_blosc(self, **kwargs):
        """ Init-function for load from blosc.

        Parameters
        ----------
        **kwargs
                src : iterable of components that need to be loaded

        Returns
        -------
        list
            list of ids of batch-items
        """
        # set images-component to 3d-array of zeroes if the component is to be updated
        if 'images' in kwargs['src']:
            self._preload_skyscraper_components(['images'])

        return self.indices

    @inbatch_parallel(init='_init_load_blosc', post='_post_default', target='async', update=False)
    async def _load_blosc(self, item_ix, *args, **kwargs):
        """ Read scans from blosc and put them into batch components

        Parameters
        ----------
        item_ix : str
            item index from batch to load 3D array
            and stack with others in images component.
        **kwargs
                 src : tuple
                     tuple of strings with names ofcomponents of data
                     that should be loaded into self

        Note
        ----
        NO conversion to HU is done for blosc
        (because usually it's done before).
        """

        for source in kwargs['src']:
            # set correct extension for each component and choose a tool
            # for debyting and (possibly) decoding it
            if source in ['spacing', 'origin']:
                ext = 'cpkl'
                unpacker = cloudpickle.loads
            else:
                ext = 'blk'
                def unpacker(byted):
                    """ Debyte and decode an ndarray
                    """
                    debyted = blosc.unpack_array(byted)

                    # read the decoder and apply it
                    decod_path = os.path.join(self.index.get_fullpath(item_ix), source, 'data.decoder')

                    # if file with decoder not exists, assume that no decoding is needed
                    if os.path.exists(decod_path):
                        with open(decod_path, mode='rb') as file:
                            decoder = cloudpickle.loads(file.read())
                    else:
                        decoder = lambda x: x

                    return decoder(debyted)

            comp_path = os.path.join(self.index.get_fullpath(item_ix), source, 'data' + '.' + ext)

            # read the component
            async with aiofiles.open(comp_path, mode='rb') as file:
                byted = await file.read()

            # de-byte it with the chosen tool
            component = unpacker(byted)

            # update needed slice(s) of component
            comp_pos = self.get_pos(None, source, item_ix)
            getattr(self, source)[comp_pos] = component

        return None

    def _load_raw(self, **kwargs):        # pylint: disable=unused-argument
        """ Load scans from .raw images (with meta in .mhd)

            Note
            ----
            NO conversion to HU is done
            NO multithreading is used, as SimpleITK (sitk) lib crashes
            in multithreading mode in our experiments.
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
    def get_linear(from_interval, to_interval):
        """ Get linear transformation that maps one interval to another

        Parameters
        ----------
        from_interval : ndarray, tuple or list
            sequence of len=2 (llim, ulim) that defines the domain-interval
        to_interval : ndarray, tuple or list
            sequence of len=2 that defines the image-interval

        Returns
        -------
        function
            linear transformation
        """
        # compute coeffs of the mapping
        llim, ulim = from_interval
        new_llim, new_ulim = to_interval
        slope = (new_ulim - new_llim) / (ulim - llim)
        intercept = new_llim - slope * llim

        # define the map
        def linear(x):
            """ Transformation
            """
            return slope * x + intercept

        return linear

    @classmethod
    async def encode_dump_array(cls, data, folder, filename, mode):
        """ Encode an ndarray to int8, blosc-pack it and dump data along with
        the decoder and shape of data into supplied folder.

        Parameters
        ----------
        cls : type
            class from which the method is executed
        data : ndarray
            contains numeric (e.g., float32) data to be dumped
        folder : str
            folder for dump
        filename : str
            name of file where the data is dumped; has format name.ext
        mode : str or None
            Mode of encoding to int8. Can be either 'quantization' or 'linear'
            or None

        Note
        ----
        currently, two modes of encoding are supported:
         - 'linear': uses linear Transformation to cast data-range to int8-range
            and then rounds off fractional part.
         - 'quantization': attempts to use histogram of pixel densities to come up with a
            transformation to int8-range that yields lesser error than linear.
        """
        # parse mode of encoding
        if isinstance(mode, int):
            if mode <= 2:
                _modes = [None, 'linear', 'quantization']
                mode = _modes[mode]
        elif isinstance(mode, str):
            mode = mode.lower()

        fname_noext = '.'.join(filename.split('.')[:-1])

        # init list of serialized objects and filenames for dump
        byted, fnames = list(), list()

        # encode the data and get the decoder
        if mode == 'linear':
            data_range = (data.min(), data.max())
            i8_range = (-128, 127)

            if data_range[0] == data_range[1]:
                value = data_range[0]
                encoded = np.zeros_like(data, dtype=np.int8)
                decoder = lambda x: x + value
            else:
                encoded = np.rint(cls.get_linear(data_range, i8_range)(data)).astype(np.int8)
                decoder = cls.get_linear(i8_range, data_range)

            # serialize decoder
            byted.append(cloudpickle.dumps(decoder))
            fnames.append(fname_noext + '.decoder')
        elif mode == 'quantization':

            # set up quantization model
            data_range = (data.min(), data.max())
            batch_size = min(KMEANS_MINIBATCH, data.size)
            model = MiniBatchKMeans(n_clusters=256, init=np.linspace(*data_range, 256).reshape(-1, 1))

            # fit the model on several minibatches, get encoded data
            for _ in range(KMEANS_ITERS):
                batch = np.random.choice(data.reshape(-1), batch_size, replace=False).reshape(-1, 1)
                model.partial_fit(batch)

            encoded = (model.predict(data.reshape(-1, 1)) - 128).astype(np.int8)

            # prepare decoder
            decoder = lambda x: (model.cluster_centers_[x + 128]).reshape(data.shape)

            # serialize decoder
            byted.append(cloudpickle.dumps(decoder))
            fnames.append(fname_noext + '.decoder')
        elif mode is None:
            encoded = data

        else:
            raise ValueError('Unknown mode of int8-encoding')

        # serialize (possibly) encoded data and its shape
        byted.extend([blosc.pack_array(encoded, cname='zstd', clevel=1), cloudpickle.dumps(np.array(data.shape))])
        fnames.extend([filename, fname_noext + '.shape'])

        # dump serialized items
        for btd, fname in zip(byted, fnames):
            async with aiofiles.open(os.path.join(folder, fname), mode='wb') as file:
                _ = await file.write(btd)


    @classmethod
    async def dump_data(cls, data_items, folder, i8_encoding_mode):
        """ Dump data from data_items on disk in specified folder

        Parameters
        ----------
        cls : type
            class from which the method is executed
        data_items : dict
            dict of data items for dump in form {item_name: [item, 'ext']}
            (e.g.: {'images': [scans, 'blk'], 'masks': [masks, 'blk'], 'spacing': [spacing, 'cpkl']})
        folder : str
            folder to dump data-items in. Note that each data item is dumped in its separate folder
        i8_encoding_mode: str, int, or dict
            contains mode of encoding to int8

        Note
        ----
        Depending on supplied format in data_items, each data-item will be either
            cloudpickle-serialized (if .cpkl) or blosc-packed (if .blk)
        """

        # create directory if does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # infer extension of each item, serialize/blosc-pack and dump the item
        for item_name, (data, ext) in data_items.items():
            item_folder = os.path.join(folder, item_name)
            if not os.path.exists(item_folder):
                os.makedirs(item_folder)
            if ext == 'blk':
                if isinstance(i8_encoding_mode, dict):
                    mode = i8_encoding_mode.get(item_name, None)
                else:
                    mode = i8_encoding_mode

                _ = await cls.encode_dump_array(data, item_folder, 'data.blk', mode)

            elif ext == 'cpkl':
                byted = cloudpickle.dumps(data)
                async with aiofiles.open(os.path.join(item_folder, 'data.cpkl'), mode='wb') as file:
                    _ = await file.write(byted)

        return None

    @action
    @inbatch_parallel(init='indices', post='_post_default', target='async', update=False)
    async def dump(self, item_ix, dst, src=None, fmt='blosc', index_to_name=None, i8_encoding_mode=None):
        """ Dump scans data (3d-array) on specified path in specified format

        Parameters
        ----------
        dst : str
            destination-folder where all patients' data should be put
        src : str or list/tuple
            component(s) that we need to dump (smth iterable or string). If not
            supplied, dump all components + shapes of scans
        fmt : 'blosc'
            format of dump. Currently only blosc-format is supported;
            in this case folder for each patient is created, patient's data
            is put into images.blk, attributes are put into files attr_name.cpkl
            (e.g., spacing.cpkl)
        index_to_name : callable or None
            returns str;
            function that relates each item's index to a name of item's folder.
            That is, each item is dumped into os.path.join(dst, index_to_name(items_index)).
            If None, no transformation is applied.
        i8_encoding_mode : int, str or dict
            whether components with .blk-format should be cast to int8-type.
            The cast allows to save space on disk and to speed up batch-loading. However,
            the cast comes with loss of precision, as originally .blk-components are stored
            in float32-format. Can be int: 0, 1, 2 or str/None: 'linear', 'quantization' or None.
            0 or None stand for no encoding. 1 stands for 'linear', 2 - for 'quantization'.
            Can also be dict of modes, e.g.: {'images': 'linear', 'masks': 0}

        Example
        -------
        Initialize batch and load data

        >>> ind = ['1ae34g90', '3hf82s76']
        >>> batch = CTImagesBatch(ind)
        >>> batch.load(...)
        >>> batch.dump(dst='./data/blosc_preprocessed')

        The command above creates following files:

        - ./data/blosc_preprocessed/1ae34g90/images/data.blk
        - ./data/blosc_preprocessed/1ae34g90/images/data.shape
        - ./data/blosc_preprocessed/1ae34g90/spacing/data.cpkl
        - ./data/blosc_preprocessed/1ae34g90/origin/data.cpkl

        - ./data/blosc_preprocessed/3hf82s76/images/data.blk
        - ./data/blosc_preprocessed/3hf82s76/images/data.shape
        - ./data/blosc_preprocessed/3hf82s76/spacing/data.cpkl
        - ./data/blosc_preprocessed/3hf82s76/origin/data.cpkl
        """
        # if src is not supplied, dump all components
        if src is None:
            src = self.components

        if fmt != 'blosc':
            raise NotImplementedError('Dump to {} is not implemented yet'.format(fmt))

        # convert src to iterable 1d-array
        src = np.asarray(src).reshape(-1)
        data_items = dict()

        # get correct extension for each component, add the component to items-dict
        for source in list(src):
            if source in ['spacing', 'origin']:
                ext = 'cpkl'
            else:
                ext = 'blk'
            # determine position in data of source-component for the patient
            comp_pos = self.get_pos(None, source, item_ix)
            data_items.update({source: [getattr(self, source)[comp_pos], ext]})

        # set item-specific folder
        item_subdir = item_ix if index_to_name is None else index_to_name(item_ix)
        item_dir = os.path.join(dst, item_subdir)

        return await self.dump_data(data_items, item_dir, i8_encoding_mode)

    def get_pos(self, data, component, index):
        """ Return a positon of an item for a given index in data
        or in self.`component`.

        Fetch correct position inside batch for an item, looks for it
        in `data`, if provided, or in `component` in self.

        Parameters
        ----------
        data :      None or ndarray
                    data from which subsetting is done.
                    If None, retrieve position from `component` of batch,
                    if ndarray, returns index.
        component : str
                    name of a component, f.ex. 'images'.
                    if component provided, data should be None.
        index :     str or int
                    index of an item to be looked for.
                    may be key from dataset (str)
                    or index inside batch (int).

        Returns
        -------
        int
            Position of item

        Note
        ----
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

    def _post_default(self, list_of_arrs, update=True, new_batch=False, **kwargs):
        """ Gatherer outputs of different workers, update `images` component

        Parameters
        ----------
        list_of_arrs : list
                       list of ndarrays to be concated and put in a batch.images.
        update :       bool
                       if False, nothing is performed.
        new_batch :    bool
                       if False, empty batch is created,
                       if True, data is gathered, loaded and put into batch.images.
        Returns
        -------
        batch
             new batch, empty batch or self-batch.

        Note
        ----
        Output of each worker should correspond to individual patient.
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
                shape :   list, tuple or ndarray
                          (z,y,x); shape of every image in image component after action is performed.
                spacing : list, tuple or ndarray
                          if supplied, assume that unify_spacing is performed

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
                item_args['res_factor'] = self.spacing[i, :] / np.array(kwargs['spacing'])
                item_args['shape_resize'] = shape_after_resize[i, :]

            all_args += [item_args]

        return all_args

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):
        """ Gather outputs of different workers for actions, rebuild `images` component.

        Parameters
        ----------
        all_outputs : list
                      list of outputs. Each item is given by tuple
        new_batch :   bool
                      if True, returns new batch with data agregated
                      from all_ouputs. if False, changes self.
        **kwargs
                shape :   list, tuple or ndarray
                          (z,y,x); shape of every image in image component after action is performed.
                spacing : list, tuple or ndarray
                          if supplied, assume that unify_spacing is performed
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

    @action
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='threads')
    def resize(self, patient, out_patient, res, shape=(128, 256, 256), method='pil-simd',
               axes_pairs=None, resample=None, order=3, *args, **kwargs):
        """ Resize (change shape of) each CT-scan in the batch.

        When called from a batch, changes this batch.

        Parameters
        ----------
        shape :      tuple
                     (z, y, x) shape that should be AFTER resize.
                     Note, that ct-scan dim_ordering also should be `z,y,x`
        method :     str
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
        resample :   filter of pil-simd resize. By default set to bilinear. Can be any of filters
                     supported by PIL.Image.
        order :      the order of scipy-interpolation (<= 5)
                     large value improves precision, but slows down the computaion.

        Example
        -------
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
    def unify_spacing(self, patient, out_patient, res, res_factor,
                      shape_resize, spacing=(1, 1, 1), shape=(128, 256, 256),
                      method='pil-simd', order=3, padding='edge', axes_pairs=None,
                      resample=None, *args, **kwargs):
        """ Unify spacing of all patients.

        Resize all patients to meet `spacing`, then crop/pad resized array to meet `shape`.

        Parameters
        ----------
        spacing :      tuple
                       (z,y,x) spacing after resize.
                       Should be passed as key-argument.
        shape :        tuple
                       (z,y,x,) shape after crop/pad.
                       Should be passed as key-argument.
        method :       str
                       interpolation method ('pil-simd' or 'resize').
                       Should be passed as key-argument.
                       See CTImagesBatch.resize for more information.
        order :        None or int
                       order of scipy-interpolation (<=5), if used.
                       Should be passed as key-argument.
        padding :      str
                       mode of padding, any supported by np.pad.
                       Should be passed as key-argument.
        axes_pairs :   tuple, list of tuples with pairs
                       pairs of axes that will be used consequentially
                       for performing pil-simd resize.
                       Should be passed as key-argument.
        resample :     None or str
                       filter of pil-simd resize.
                       Should be passed as key-argument
        patient :      str
                       index of patient, that worker is handling.
                       Note: this argument is passed by inbatch_parallel
        out_patient :  ndarray
                       result of individual worker after action.
                       Note: this argument is passed by inbatch_parallel
        res         :  ndarray
                       New `images` to replace data inside `images` component.
                       Note: this argument is passed by inbatch_parallel
        res_factor  :  tuple
                       (float), factor to make resize by.
                       Note: this argument is passed by inbatch_parallel
        shape_resize : tuple
                       It is possible provide `shape_resize` (shape after resize)
                       instead of spacing. Then array with `shape_resize`
                       will be cropped/padded for shape to = `shape` arg.
                       Note: this argument is passed by inbatch_parallel


        Note
        ----
        see CTImagesBatch.resize for more info about methods' params.

        Example
        -------
        >>> shape = (128, 256, 256)
        >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                        order=2, method='scipy', padding='reflect')
        >>> batch = batch.unify_spacing(shape=shape, spacing=(1.0, 1.0, 1.0),
                                        resample=PIL.Image.BILINEAR)
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
    @inbatch_parallel(init='indices', post='_post_default', update=False, target='threads')
    def rotate(self, index, angle, components='images', axes=(1, 2), random=True, **kwargs):
        """ Rotate 3D images in batch on specific angle in plane.

        Parameters
        ----------
        angle :      float
                     degree of rotation.
        components : list, tuple or str
                     name(s) of components to rotate each item in it.
        axes :       tuple
                     (int, int), plane of rotation specified by two axes (zyx-ordering).
        random :     bool
                     if True, then degree specifies maximum angle of rotation.
        index :      int
                     index of patient in batch.
                     This argument is passed by inbatch_parallel

        Returns
        -------
        ndarray
                ndarray of 3D rotated image.

        Note
        ----
        zero padding automatically added after rotation.

        Example
        -------
        Rotate images on 90 degrees:

        >>> batch = batch.rotate(angle=90, axes=(1, 2), random=False)

        Random rotation with maximum angle:

        >>> batch = batch.rotate(angle=30, axes=(1, 2))

        """
        if not isinstance(components, (tuple, list)):
            _components = (components, )

        _angle = angle * np.random.rand() if random else angle
        for comp in _components:
            data = self.get(index, comp)
            rotate_3D(data, _angle, axes)

    @action
    @inbatch_parallel(init='_init_images', post='_post_default', target='nogil', new_batch=True)
    def make_xip(self, step=2, depth=10, func='max', projection='axial', *args, **kwargs):
        """ Make intensity projection (maximum, minimum, average)

        Popular radiologic transformation: max, min, avg applyied along an axe.
        Notice that axe is chosen in accordance with projection argument.

        Parameters
        ----------
        step :       int
                     stride-step along axe, to apply the func.
        depth :      int
                     depth of slices (aka `kernel`) along axe made on each step for computing.
        func :       str
                     Possible values are 'max', 'min' and 'avg'.
        projection : str
                     Possible values: 'axial', 'coroanal', 'sagital'.
                     In case of 'coronal' and 'sagital' projections tensor
                     will be transposed from [z,y,x] to [x, z, y] and [y, z, x].

        Returns
        -------
        ndarray
               resulting ndarray after func is applied.

        """
        return xip_fn_numba(func, projection, step, depth)

    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='nogil', new_batch=True)
    def calc_lung_mask(self, *args, **kwargs):     # pylint: disable=unused-argument, no-self-use
        """ Return a mask for lungs """
        return calc_lung_mask_numba

    @action
    def segment(self, erosion_radius=2):
        """ Segment lungs' content from 3D array.

        Paramters
        ---------
        erosion_radius : int
                         radius of erosion to be performed.

        Note
        ----
        Sets HU of every pixel outside lungs to DARK_HU = -2000.

        Example
        -------

        >>> batch = batch.segment(erosion_radius=4, num_threads=20)
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
        """ Make crop of crop_size from center of images.

        Parameters
        ----------
        crop_size : tuple
                    (int, int, int), size of crop, in `z,y,x`.
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
        """ Extract patches of patch_shape with specified stride.

        Parameters
        ----------
        patch_shape : tuple, list or ndarray
                      (z_dim,y_dim,x_dim), shape of a single patch.
        stride :      tuple, list or ndarray
                      (int, int, int), stride to slide over each patient's data.
        padding :      str
                      padding-type (see doc of np.pad for available types).
        data_attr :    str
                      component to get data from.

        Returns
        -------
        ndarray
                4d-ndaray of patches; first dimension enumerates patches

        Note
        ----
        Shape of all patients data is needed to be the same at this step,
        resize/unify_spacing is required before.
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
        """ Get skyscraper from 4d-array of patches, put it to `data_attr` component in batch.

        Let reconstruct original skyscraper from patches (if same arguments are passed)

        Parameters
        ----------
        patches :    ndarray
                     4d-array of patches, with dims: `(patch_nb, z, y, x)`.
        scan_shape : tuple, list or ndarray
                     (z,y,x), shape of individual scan (should be same for all scans).
        stride :     tuple, list or ndarray
                     stride-step used for gathering data from patches.
        data_attr :  str
                     batch component name to store new data.

        Note
        ----
        If stride != patch.shape(), averaging of overlapped regions is used.
        `scan_shape`, patches.shape(), `stride` are used to infer the number of items
        in new skyscraper. If patches were padded, padding is removed for skyscraper.

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
        """ Normalize HU-densities to interval [0, 255].

        Trim HU that are outside range [min_hu, max_hu], then scale to [0, 255].

        Parameters
        ----------
        min_hu : int
                 minimum value for hu that will be used as trimming threshold.
        max_hu : int
                 maximum value for hu that will be used as trimming threshold.

        Example
        -------
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
    @inbatch_parallel(init='_init_rebuild', post='_post_rebuild', target='nogil')
    def flip(self):    # pylint: disable=no-self-use
        """ Invert the order of slices for each patient

        Example
        -------
        >>> batch = batch.flip()
        """
        return flip_patient_numba

    def get_axial_slice(self, person_number, slice_height):
        """ Get axial slice (e.g., for plots)

        Parameters
        ----------
        person_number : str or int
                        Can be either index (int) of person in the batch
                        or patient_id (str)
        slice_height :  float
                        scaled from 0 to 1 number of slice.
                        e.g. 0.7 means that we take slice with number
                        int(0.7 * number of slices for person)

        Example
        -------
        Here self.index[5] usually smth like 'a1de03fz29kf6h2'

        >>> patch = batch.get_axial_slice(5, 0.6)
        >>> patch = batch.get_axial_slice(self.index[5], 0.6)

        """
        margin = int(slice_height * self.get(person_number, 'images').shape[0])
        patch = self.get(person_number, 'images')[margin, :, :]
        return patch
