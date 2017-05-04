"""Contains class CTImagesMaskedBatch for storing masked Ct-scans."""

from concurrent.futures import ThreadPoolExecutor
from binascii import hexlify
from collections import namedtuple
import numpy as np
from numba import jit
import SimpleITK as sitk
from .ct_batch import CTImagesBatch
# from .mask import make_mask_patient, insert_cropped
from .resize import resize_patient_numba
from .dataset_import import action
from .dataset_import import inbatch_parallel


NoduleBase = namedtuple(typename='NoduleBase',
                        field_names=['patient_id', 'z',
                                     'y', 'x', 'd'])


class Nodule(NoduleBase):
    """Extended Nodule class."""

    def mask(self, spacing):
        """Get 3d numpy array with mask.

        Returns ndarray(Nz, Ny, Nx) array filled with 1 and 0
        representing the mask of the nodule.

        Args:
        - spacing: list, tuple or numpy array with spacing;
        """
        return np.ones(self.size / spacing, dtype=np.int32)

    def start_pix(self, origin, spacing):
        """Get the start pixel of the Nodule.

        Returns ndarray(3, ) with coordinate of nodule's start
        pixel in patient's data array.

        Args:
        - origin: list, tuple or numpy array with origin coordinates;
        - spacing: list, tuple or numpy array with spacing;
        """
        origin = np.asarray(origin)
        spacing = np.asarray(spacing)
        start_pix = np.abs(self.center - origin) - (self.size / 2)
        return np.rint(start_pix / spacing).astype(np.int32)

    def center_pix(self, origin, spacing):
        """Get the center pixel of the Nodule.

        Returns ndarray(3, ) with coordinates of nodule's center
        pixel in patient's data array.

        Args:
        - origin: list, tuple or numpy array with origin coordinates;
        - spacing: list, tuple or numpy array with spacing;
        """
        origin = np.asarray(origin)
        spacing = np.asarray(spacing)
        center_pix = np.abs(self.center - origin)
        return np.rint(center_pix / spacing).astype(np.int32)

    def size_pix(self, spacing):
        """Get nodule's pixel sizes.

        Returns ndarray(3, ) with nodule's sizes in number of pixels.

        Args:
        - spacing: list, tuple of numpy array with spacing;
        """
        spacing = np.asarray(spacing)
        return np.rint(self.size / spacing).astype(np.int32)

    @property
    def start(self):
        """Get nodule's start position in world's coordinates.

        This property returns ndarray(3, ) with nodule's
        start position in world's coordinate system.
        """
        return self.center - (self.size / 2)

    @property
    def center(self):
        """Get nodule's start position in world's coordinates.

        This property returns ndarray(3, ) with nodule's
        start position in world's coordinate system.
        """
        coords = np.array([self.z, self.y, self.x])
        return coords

    @property
    def size(self):
        """Get nodule's absolute sizes.

        This property returns ndarray(3, ) with nodule's
        absolute sizes computed in world's cordinate system.
        """
        return np.tile(self.d, 3)


@jit('float64[:, :, :](float64[:, :, :], int32[:, :], int32[:])', nogil=True)
def get_nodules_jit(data: "ndarray(l, j, k)",
                    positions: "ndarray(q, 3)", size: "ndarray(3, )"):
    """Fetch nodules from array by array of starting positions.

    This numberized function takes source array with data of shape (n, k, l)
    represented by 3d numpy array with BatchCt data,
    ndarray(p, 3) with starting indices of nodules where p is number
    of nodules and size of type ndarray(3, ) which contains
    sizes of nodules along each axis. The output is 3d ndarray with nodules
    put in CTImagesBatch-compatible skyscraper structure.

    *Note that dtypes of positions and size arrays must be the same.

    Args:
    - data: CTImagesBatch skyscraper represented by 3d numpy array;
    - positions: ndarray(l, 3) of int32 containing
      nodules' starting indices along [zyx]-axis
      accordingly in ndarray data;
    - size: ndarray(3,) of int32 containing
      nodules' sizes along each axis;
    """
    out_arr = np.zeros((positions.shape[0], size[0],
                        size[1], size[2]), dtype=data.dtype)

    n_positions = positions.shape[0]
    for i in range(n_positions):
        out_arr[i, :, :, :] = data[positions[i, 0]: positions[i, 0] + size[0],
                                   positions[i, 1]: positions[i, 1] + size[1],
                                   positions[i, 2]: positions[i, 2] + size[2]]

    return out_arr.reshape(n_positions * size[0], size[1], size[2])


class CTImagesMaskedBatch(CTImagesBatch):
    """Class for storing masked batch of ct-scans.

    In addition to batch itself, stores mask in
    self.mask as ndarray, origin and spacing dictionaries
    and list with information about nodules in batch.

    new attrs:
        1. mask: ndarray of masks
        2. spacing: dict with keys = self.indices
            stores distances between pixels in mm for patients
            order is x, y, z
        3. origin: dict with keys = self.indices
            stores world coords of [0, 0, 0]-pixel of data for
            all patients
        4. nodules_info: list with information about nodule; each nodule
            represented by instance of Nodule class

    Important methods:
        1. load_mask(self, nodules_df, num_threads=8)
            function for
            loading masks from dataframe with nodules
            multithreading is supported
        2. resize(self, num_x_new=256, num_y_new=256,
                  num_slices_new=128, order=3, num_threads=8)
            transform shape of all patients to
            (num_slices_new, num_y_new, num_x_new)
            if masks are loaded, they are are also resized

        *Note: spacing, origin are recalculated when resize is executed
            As a result, load_mask can be also executed after resize
    """

    @staticmethod
    def make_indices(size: 'int'):
        """Generate list of batch indices of given size.

        Take number of indices as input parameter size and
        generates list of random indices of length size.

        Args:
        - size: size of list with indices;
        """
        random_data = np.random.uniform(0, 1, size=(size, 10)) * 123456789
        indices = [hexlify(random_data[i, :])[:8].decode("utf-8")
                   for i in range(size)]
        return indices

    @staticmethod
    def dump_blosc(data, index, path):
        """Dump data on hard disk in blosc format.

        Save data on hard drive in file with path
        os.path.join(path, str(index)).
        """
        full_path = os.path.join(path, index)
        packed = blosc.pack_array(data, cname='zstd', clevel=1)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
        os.makedirs(full_path)
        with open(os.path.join(full_path, 'data.blk'), mode='wb') as dump_file:
            dump_file.write(packed)

    def __init__(self, index):
        """Initialization of CTImagesMaskedBatch.

        Initialize CTImagesMaskedBatch with index.
        """
        super().__init__(index)
        self.mask = None
        self.nodules_info = None
        self.spacing = dict()
        self.origin = dict()

    @action
    def load(src=None, fmt='dicom', bounds=None,
             origin=None, spacing=None, nodules_info=None):
        """Load data in masked batch of patients.

        Args:
        - src: source array with skyscraper, needed if fmt is 'ndarray';
        - fmt: type of source data; possible values are 'raw' and 'ndarray';
        Returns:
        - self;

        Examples:
        >>> index = FilesIndex(path="/some/path/*.mhd, no_ext=True")
        >>> batch = CTImagesMaskedBatch(index)
        >>> batch.load(fmt='raw')

        >>> batch.load(src=source_array, fmt='ndarray', bounds=bounds,
        ...            origin=origin_dict, spacing=spacing_dict)
        """
        if fmt == 'raw':
            self._load_raw()
        elif fmt in ['dicom', 'blosc']:
            raise NotImplementedError("This load format option " +
                                      "is not implemented for masked batch")
        elif fmt == 'ndarray':
            self._data = src
            self._bounds = bounds
            self.origin = origin
            self.spacing = spacing
        else:
            raise TypeError("Incorrect type of batch source")

    @inbatch_parallel(init='indices', post='_post_default', target='threads')
    def _load_raw(self, patient_id, *args, **kwargs):
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
        self.origin[patient_id] = np.array(raw_data.GetOrigin())[::-1]
        self.spacing[patient_id] = np.array(raw_data.GetSpacing())[::-1]
        return sitk.GetArrayFromImage(raw_data)

    def get_mask(self, index):
        """Get view on patient data's mask.

        This method takes position of patient in self or his index
        and returns view on patient data's mask.

        Args:
        - index: can be either position of patient in self._data
        or index from self.index;

        Return:
        - ndarray(Nz, Ny, Nz): view on patient data's mask array;
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
        return self.mask[lower: upper, :, :]

    @action
    def fetch_nodules_info(self, nodules_df, update=False):
        """Get nodules in 3d ndarray.

        This method fetch info about all nodules in batch
        and put them in 2d numpy array.
        """
        if self.nodules_info is not None and not update:
            return self
        nodules_df = nodules_df.set_index('seriesuid')

        unique_indices = nodules_df.index.unique()
        inter_index = list(set(self.indices) & set(unique_indices))
        # TODO enable indices intersection via numpy
        nodules_df = nodules_df.loc[inter_index,
                                    ["coordZ", "coordY",
                                     "coordX", "diameter_mm"]]

        self.nodules_info = []
        for pat_id, coordz, coordy, coordx, diam in nodules_df.itertuples():
            nodule = Nodule(pat_id, coordz, coordy, coordx, diam)
            self.nodules_info.append(nodule)
        return self

    def _nodule_bounds(self, nodule, nodule_size=None):
        """Transform nodule world coordinates to pixel coords of patient.

        Args:
        - nodule: nodule instance;
        location in patient data;
        Return:
        - ndarray(, 3) pixel coordinates of nodule in batch skyscraper;
        """
        patient_pos = self.index.get_pos(nodule.patient_id)
        origin = self.origin[nodule.patient_id]
        spacing = self.spacing[nodule.patient_id]

        if nodule_size is None:
            nodule_size = nodule.size_pix(spacing)

        img_size = np.array(self.get_image(nodule.patient_id).shape,
                            dtype='int32')
        start_pix = nodule.start_pix(origin, spacing)
        end_pix = start_pix + nodule_size

        bias_out_of_bounds = np.where(end_pix > img_size,
                                      end_pix - img_size, np.zeros(3))
        start_pix -= bias_out_of_bounds
        end_pix -= bias_out_of_bounds

        patient_bias = np.array([self._bounds[patient_pos], 0, 0],
                                dtype=np.int32)

        start_pix += patient_bias
        end_pix += patient_bias
        return start_pix

    @action
    def create_mask(self):
        """Load mask data for using nodule's info.

        Load mask into self.mask using info in attribute self.nodules_info.
        *Note: nodules info must be loaded before the call of this method.
        """
        if self.nodules_info is None:
            raise AttributeError("Info about nodules location must " +
                                 "be loaded before calling this method")
        for nodule in self.nodules_info:
            mask_view = self.get_mask(nodule.patient_id)
            origin = self.origin[nodule.patient_id]
            spacing = self.spacing[nodule.patient_id]

            size = nodule.size_pix(spacing)
            start = nodule.start_pix(origin, spacing)
            end = np.minimum(start + size, np.asarray(mask_view.shape))

            if np.any(start < 0):
                raise ValueError("Start pixel of nodule has negative " +
                                 "coordinates")

            mask_view[start[0]: end[0],
                      start[1]: end[1],
                      start[2]: end[2]] = nodule.mask(spacing)

        return self

    def sample_random_nodules(self, n_nodules: 'int',
                              nodule_size: 'ndarray(3, )') -> "ndarray(l, 3)":
        """Sample random nodules from CTImagesBatchMasked skyscraper.

        Samples random n_nodules' lower_bounds coordinates
        and stack obtained data into ndarray(l, 3) then returns it.
        First dimension of that array is just an index of sampled
        nodules while second points out pixels of start of nodules
        in BatchCt skyscraper. Each nodule have shape
        defined by parameter size. If size of patients' data along
        z-axis is not the same for different patients than
        NotImplementedError will be raised.

        Args:
        - n_nodules: number of random nodules to sample from BatchCt data;
        - nodule_size: ndarray(3, ) nodule size in number of pixels;

        return
        - ndarray(l, 3) of int that contains information
        about starting positions
        of sampled nodules in BatchCt skyscraper along each axis.
        First dimension is used to index nodules
        while the second one refers to various axes.

        *Note: [zyx]-ordering is used;
        """
        all_indices = np.arange(len(self.indices))
        sampled_indices = np.random.choice(all_indices,
                                           n_nodules, replace=True)

        shape_z = (self._bounds[sampled_indices] -
                   self._bounds[sampled_indices + 1])
        shape_z = shape_z.reshape(-1, 1)

        offset = np.vstack([np.asarray(self._bounds[sampled_indices]),
                            np.zeros(n_nodules), np.zeros(n_nodules)]).T

        shapes_yx = np.tile([self.data.shape[1],
                             self.data.shape[2]], n_nodules)

        shapes_yx = shapes_yx.reshape(-1, 2)

        data_shape = np.concatenate([shape_z, shapes_yx], axis=1)
        samples = np.random.rand(n_nodules, 3) * (data_shape - nodule_size)
        return samples + offset

    @action
    def sample_nodules(self, batch_size,
                       nodule_size, share=0.8) -> 'CTImagesBatchMasked':
        """Fetch random cancer and non-cancer nodules from batch.

        Fetch nodules from CTImagesBatchMasked into ndarray(l, m, k).

        Args:
        - nodules_df: dataframe of csv file with information
        about nodules location;
        - batch_size: number of nodules in the output batch. Must be int;
        - nodule_size: size of nodule along axes.
        Must be list, tuple or nsystem pathdarray(3, ) of integer type;
        (Note: using zyx ordering)
        - share: share of cancer nodules in the batch.
        If source CTImagesBatch contains less cancer
        nodules than needed random nodules will be taken;
        """
        if self.nodules_info is None:
            raise AttributeError("Info about nodules location must " +
                                 "be loaded before calling this method")

        nodule_size = np.asarray(nodule_size, dtype=np.int32)

        nodules = np.stack([self._nodule_bounds(nodule)
                            for nodule in self.nodules_info])

        cancer_n = int(share * batch_size)
        cancer_n = (nodules.shape[0] if cancer_n > nodules.shape[0]
                    else cancer_n)
        if nodules.shape[0] == 0:
            cancer_nodules = np.zeros((0, 3))
        else:
            sample_indices = np.random.choice(np.arange(nodules.shape[0]),
                                              size=cancer_n, replace=False)
            cancer_nodules = nodules[sample_indices, :]

        random_nodules = self.sample_random_nodules(batch_size - cancer_n,
                                                    nodule_size)
        nodules_indices = np.vstack([cancer_nodules,
                                     random_nodules]).astype('int32')  # pylint: disable=no-member

        data = get_nodules_jit(self.data, nodules_indices, nodule_size)
        mask = get_nodules_jit(self.mask, nodules_indices, nodule_size)
        bounds = np.arange(data.shape[0] + 1) * nodule_size[0]

        nodules_batch = CTImagesMaskedBatch(self.make_indices(batch_size))
        nodules_batch.load(src=data, fmt='ndarray', upper_bounds=bounds)
        nodules_batch.mask = mask
        nodules_batch.origin = self.origin
        nodules_batch.spacing = self.spacing
        return nodules_batch

    @action
    def dump(self, dst, fmt='blosc', dtype='source'):
        """Dump patients data and mask(optional) on disc.

        Dump on specified path and format
        create folder corresponding to each patient
        *Note: this method is decorated with @history and @action.
        If mask_dst in not None than dump mask too.

        Example:
        # initialize batch and load data
        >>> ind = ['1ae34g90', '3hf82s76', '2ds38d04']
        >>> batch = BatchCt(ind)
        >>> batch.load(...)
        >>> batch.dump('./data/blosc_preprocessed', dtype='source')

        # the command above creates files
        # ./data/blosc_preprocessed/1ae34g90/data.blk
        # ./data/blosc_preprocessed/3hf82s76/data.blk
        # ./data/blosc_preprocessed/2ds38d04/data.blk
        """
        dtype_values = ['source', 'mask']

        if isinstance(dtype, (tuple, list)):
            if any(dt not in dtype_values for dt in dtype):
                raise ValueError("Argument dtype must be list or tuple" +
                                 "containing 'source' or 'mask'")
            if len(dtype) != len(dst):
                raise ValueError("Arguments dtype and dst must have " +
                                 "the same length if having " +
                                 "type list or tuple")

        elif not(isinstance(dtype, str) and isinstance(dst, str)):
            raise ValueError("Arguments dtype and dst must " +
                             "have the same type")

        for patient_id in self.indices:
            patient_pos = self.index.get_pos(patient_id)
            lower = self._bounds[patient_pos]
            upper = self._bounds[patient_pos + 1]

            for dump_type, dump_path in zip(dtype, dst):
                if dump_type == 'source':
                    dump_blosc(self.data[lower: upper, :, :],
                               patient_id, dump_path)
                elif dump_type == 'mask':
                    dump_blosc(self.mask[lower: upper, :, :],
                               patient_id, dump_path)
        return self

    def get_axial_slice(self, patient_pos, height):
        """Get tuple of slices (data slice, mask slice).

        Args:
            patient_pos: patient position in the batch
            height: height, take slices with number
                int(0.7 * number of slices for patient) from
                patient's scan and mask
        """
        margin = int(slice_height * self[patient_pos].shape[0])

        patch = (self.get_image(patient_pos)[margin, :, :],
                 self.get_mask(patient_pos)[margin:, :, :])
        return patch

    def _rescale_spacing(self, new_shape):
        """Rescale spacing during resize.

        During resize action it is neccessary to update patient's
        current spacing cause it used for mask creation
        and nodules extraction.

        Args:
        - new_shape: list, tuple or ndarray(3, ) that represents
        new_shape of patient's scans;
        Returns:
        - self;
        """
        for patient_id in self.indices:
            old_shape = np.asarray(self.get_image(patient_id).shape)
            new_shape = np.asarray(new_shape)
            self.spacing[patient_id] *= (old_shape / new_shape)
        return self

    def _init_rebuild(self, **kwargs):
        """Args-fetcher for resize parallelization.

        args-fetcher for parallelization using decorator
            can be used when batch-data is rebuild from scratch
        if shape is supplied as one of the args
            assumes that data should be resizd
        """
        if 'shape' is not in kwargs:
            raise TypeError("Output shape must be" +
                            "specified in argument shape!")
        self._rescale_spacing(new_shape=shape)
        return super()._init_rebuild(**kwargs)

    def _post_rebuild(self, workers_outputs, new_batch=False, **kwargs):
        """Post-function for resize parallelization.

        gatherer of outputs from different workers for
            ops, requiring complete rebuild of batch._data
        args:
            new_batch: if True, returns new batch with data
                agregated from workers_ouputs
        """
        # TODO: process errors
        batch = super()._post_rebuild(all_outputs, new_batch, **kwargs)
        batch.origin = self.origin
        batch.spacing = self.spacing
        batch.nodules_info = self.nodules.info
        batch.create_mask()
        return batch
