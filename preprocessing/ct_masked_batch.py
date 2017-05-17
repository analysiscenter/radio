# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=arguments-differ
"""Contains class CTImagesMaskedBatch for storing masked Ct-scans."""
import os
from binascii import hexlify
import logging
import shutil
import blosc
import numpy as np
from numba import njit
import SimpleITK as sitk
from .ct_batch import CTImagesBatch
from .mask import make_mask_patient
from .mask import make_mask
from .resize import resize_patient_numba
from .dataset_import import action
from .dataset_import import inbatch_parallel
from .dataset_import import any_action_failed


LOGGING_FMT = (u"%(filename)s[LINE:%(lineno)d]#" +
               "%(levelname)-8s [%(asctime)s]  %(message)s")
logging.basicConfig(format=LOGGING_FMT, level=logging.DEBUG)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@njit(nogil=True)
def get_nodules_jit(data, positions, size):
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
    - positions: ndarray(l, 3) of int containing
      nodules' starting indices along [zyx]-axis
      accordingly in ndarray data;
    - size: ndarray(3,) of int containing
      nodules' sizes along each axis;
    """
    out_arr = np.zeros((np.int(positions.shape[0]), size[0], size[1], size[2]))

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

    nodules_dtype = np.dtype([('patient_pos', np.int, 1),
                              ('bias', np.int, (3,)),
                              ('img_size', np.int, (3,)),
                              ('center', np.float, (3,)),
                              ('nod_size', np.float, (3,)),
                              ('spacing', np.float, (3,)),
                              ('origin', np.float, (3,))])

    @staticmethod
    def make_indices(size):
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

    @staticmethod
    def normal_shift3d(n_samples, shift_sigma=(3, 3, 3)):
        """Generate ndarray(n_samples, 3) of random shifts.

        This static method generates array of random shifts
        for samples(nodules). Shifts along each axis are normaly
        distributed with
        sigma = [shigt_sigma[0], shift_sigma[1], shift_sigma[2]]
        for shift along [z, y, x] axes correspondingly.
        """
        shft_z = np.random.normal(scale=shift_sigma[0], size=n_samples)
        shft_y = np.random.normal(scale=shift_sigma[1], size=n_samples)
        shft_x = np.random.normal(scale=sfigt_sigma[2], size=n_samples)
        return np.stack([shft_z, shft_y, shft_x]).T

    def __init__(self, index):
        """Initialization of CTImagesMaskedBatch.

        Initialize CTImagesMaskedBatch with index.
        """
        super().__init__(index)
        self.mask = None
        # record array contains the following information about nodules:
        # - self.nodules.center -- ndarray(n_nodules, 3) centers of
        #   nodules in world coords;
        # - self.nodules.nod_size -- ndarray(n_nodules, 3) sizes of
        #   nodules along z, y, x in world coord;
        # - self.nodules.img_size -- ndarray(n_nodules, 3) sizes of images of
        #   patient data corresponding to nodules;
        # - self.nodules.bias -- ndarray(n_nodules, 3) of biases of
        #   patients which correspond to nodules;
        # - self.nodules.spacing -- ndarray(n_nodules, 3) of spacinf attribute
        #   of patients which correspond to nodules;
        # - self.nodules.origin -- ndarray(n_nodules, 3) of origin attribute
        #   of patients which correspond to nodules;
        self.nodules = None

    @action
    def load(self, source=None, fmt='dicom', bounds=None,
             origin=None, spacing=None, nodules=None, mask=None):
        """Load data in masked batch of patients.

        Args:
        - source: source array with skyscraper, needed if fmt is 'ndarray';
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
        if fmt == 'ndarray':
            self._init_data(source=source, bounds=bounds,
                            origin=origin, spacing=spacing)
            self.nodules = nodules
            self.mask = mask
        else:
            super().load(source=source, bounds=bounds,
                         origin=origin, spacing=spacing, fmt=fmt)
        return self

    @action
    @inbatch_parallel(init='indices', post='_post_default',
                      target='async', update=False)
    async def dump(self, patient, dst, src='mask', fmt='blosc'):
        """Dump mask or source data on specified path and format.mro

        Dump data or mask in CTIMagesMaskedBatch on specified path and format.
        Create folder corresponing to each patient.

        example:
            # initialize batch and load data
            ind = ['1ae34g90', '3hf82s76', '2ds38d04']
            batch.load(...)
            batch.create_mask(...)
            batch.dump(dst='./data/blosc_preprocessed', src='data')
            # the command above creates files

            # ./data/blosc_preprocessed/1ae34g90/data.blk
            # ./data/blosc_preprocessed/3hf82s76/data.blk
            # ./data/blosc_preprocessed/2ds38d04/data.blk
            batch.dump(dst='./data/blosc_preprocessed_mask', src='mask')
        """
        if fmt != 'blosc':
            raise NotImplementedError('Dump to {} is not implemented yet'.format(fmt))
        if src == 'data':
            data_to_dump = self.get_image(patient)
        elif src == 'mask':
            data_to_dump = self.get_mask(patient)
        return await self.dump_blosc_async(data_to_dump)

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
        if self.mask is None:
            return None

        if isinstance(index, int):
            if index < self.batch_size and index >= 0:
                pos = index
            else:
                raise IndexError("Index is out of range")
        else:
            pos = self.index.get_pos(index)

        return self.mask[self.lower_bounds[pos]: self.upper_bounds[pos], :, :]

    @property
    def n_nodules(self):
        """Get number of nodules in CTImagesMaskedBatch.

        This property returns the number
        of nodules in CTImagesMaskedBatch. If fetch_nodules_info
        method has not been called yet returns -1.
        """
        if self.nodules is not None:
            return self.nodules.patient_pos.shape[0]
        else:
            return 0

    @action
    def fetch_nodules_info(self, nodules_df, update=False):
        """Extract nodules' info from nodules_df into attribute self.nodules.

        This method fetch info about all nodules in batch
        and put them in numpy record array which can be accessed outside
        the class by self.nodules. Record array self.nodules
        has 'spacing', 'origin', 'img_size' and 'bias' properties, each
        represented by ndarray(n_nodules, 3) referring to spacing, origin,
        image size and bound of patients which correspond to fetched nodules.
        Record array self.nodules also contains attributes 'center' and 'size'
        which contain information about center and size of nodules in
        world coordinate system, each of these properties is represented by
        ndarray(n_nodules, 3). Finally, self.nodules.patient_pos refers to
        positions of patients which correspond to stored nodules.
        Object self.nodules is used by some methods which create mask
        or sample nodule batch to perform transform from world coordinate
        system to pixel one.
        """
        if self.nodules is not None and not update:
            logger.warning("Nodules have already been extracted. " +
                           "Put update argument as True for refreshing")
            return self
        nodules_df = nodules_df.set_index('seriesuid')

        unique_indices = nodules_df.index.unique()
        inter_index = np.intersect1d(unique_indices, self.indices)
        nodules_df = nodules_df.loc[inter_index,
                                    ["coordZ", "coordY",
                                     "coordX", "diameter_mm"]]

        n_nodules = nodules_df.shape[0]
        self.nodules = np.rec.array(np.zeros(n_nodules,
                                             dtype=self.nodules_dtype))
        counter = 0
        for pat_id, coordz, coordy, coordx, diam in nodules_df.itertuples():
            pat_pos = self.index.get_pos(pat_id)
            self.nodules.patient_pos[counter] = pat_pos
            self.nodules.center[counter, :] = np.array([coordz,
                                                        coordy,
                                                        coordx])
            self.nodules.nod_size[counter, :] = np.array([diam, diam, diam])
            counter += 1

        self._refresh_nodules_info()
        return self

    def _shift_out_of_bounds(self, size, shift_scale=None):
        """Fetch start pixel coordinates of all nodules.

        This method returns start pixel coordinates of all nodules
        in batch. Note that all nodules are considered to have the
        fixed size defined by argument size: if nodule is out of
        patient's 3d image bounds than it's center is shifted.

        Args:
        - size: list, tuple of numpy array of length 3 with pixel
        size of nodules.
        """
        size = np.array(size, dtype=np.int)

        center_pix = np.abs(self.nodules.center -
                            self.nodules.origin) / self.nodules.spacing
        start_pix = (np.rint(center_pix) - np.rint(size / 2))
        if shift_scale is not None:
            start_pix += self.normal_shift3d(self.n_nodules, shift_scale)
        end_pix = start_pix + size

        bias_upper = np.maximum(end_pix - self.nodules.img_size, 0)
        start_pix -= bias_upper
        end_pix -= bias_upper

        bias_lower = np.maximum(-start_pix, 0)
        start_pix += bias_lower
        end_pix += bias_lower

        return (start_pix + self.nodules.bias).astype(np.int)

    @action
    def create_mask(self):
        """Load mask data for using nodule's info.

        Load mask into self.mask using info in attribute self.nodules_info.
        *Note: nodules info must be loaded before the call of this method.
        """
        if self.nodules is None:
            logger.warning("Info about nodules location must " +
                           "be loaded before calling this method. " +
                           "Nothing happened.")
        self.mask = np.zeros_like(self.data)

        center_pix = np.abs(self.nodules.center -
                            self.nodules.origin) / self.nodules.spacing
        start_pix = (np.rint(center_pix) - np.rint(self.nodules.nod_size / 2))

        make_mask(self.mask, self.nodules.bias,
                  self.nodules.img_size + self.nodules.bias,
                  start_pix, self.nodules.nod_size)

        return self

    def sample_random_nodules(self, n_nodules, nodule_size):
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
        all_indices = np.arange(self.batch_size)
        sampled_indices = np.random.choice(all_indices,
                                           n_nodules, replace=True)

        offset = np.zeros((n_nodules, 3))
        offset[:, 0] = self.lower_bounds

        data_shape = self.shape[sampled_indices, :]
        samples = np.random.rand(n_nodules, 3) * (data_shape - nodule_size)
        return np.asarray(samples + offset, dtype=np.int)

    @action
    def sample_nodules(self, batch_size,
                       nodule_size, share=0.8, scale=None) -> 'CTImagesBatchMasked':
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
        if self.nodules is None:
            raise AttributeError("Info about nodules location must " +
                                 "be loaded before calling this method")
        if scale is not None:
            scale = np.asarray(scale, dtype=np.int)
            scale = scale.flatten()
            if len(scale) != 3:
                logger.warning('Argument scale be np.array-like' +
                               'and has shape (3,). ' +
                               'Would be used no-scale-shift.')
                scale = None
        nodule_size = np.asarray(nodule_size, dtype=np.int)
        cancer_n = int(share * batch_size)
        cancer_n = self.n_nodules if cancer_n > self.n_nodules else cancer_n
        if self.n_nodules == 0:
            cancer_nodules = np.zeros((0, 3))
        else:
            sample_indices = np.random.choice(np.arange(self.n_nodules),
                                              size=cancer_n, replace=False)
            cancer_nodules = self._shift_out_of_bounds(nodule_size,
                                                       shift_scale=scale)
            cancer_nodules = cancer_nodules[sample_indices, :]

        random_nodules = self.sample_random_nodules(batch_size - cancer_n,
                                                    nodule_size)

        nodules_indices = np.vstack([cancer_nodules,
                                     random_nodules]).astype(np.int)  # pylint: disable=no-member

        data = get_nodules_jit(self.data, nodules_indices, nodule_size)
        mask = get_nodules_jit(self.mask, nodules_indices, nodule_size)
        bounds = np.arange(batch_size + 1) * nodule_size[0]

        nodules_batch = CTImagesMaskedBatch(self.make_indices(batch_size))
        nodules_batch.load(src=data, fmt='ndarray',
                           bounds=bounds, spacing=self.spacing)
        #TODO add info about nodules by changing self.nodules
        nodules_batch.mask = mask
        return nodules_batch

    def get_axial_slice(self, patient_pos, height):
        """Get tuple of slices (data slice, mask slice).

        Args:
            patient_pos: patient position in the batch
            height: height, take slices with number
                int(0.7 * number of slices for patient) from
                patient's scan and mask
        """
        margin = int(height * self[patient_pos].shape[0])
        if self.mask is not None:
            patch = (self.get_image(patient_pos)[margin, :, :],
                     self.get_mask(patient_pos)[margin:, :, :])
        else:
            patch = (self.get_image(patient_pos)[margin, :, :], None)
        return patch

    def _refresh_nodules_info(self):
        """Refresh self.nodules attributes [spacing, origin, img_size, bias].

        This method should be called when it is needed to make
        [spacing, origin, img_size, bias] attributes of self.nodules
        to correspond the structure of batch's inner data.
        """
        self.nodules.bias[:, 0] = self.lower_bounds[self.nodules.patient_pos]
        self.nodules.spacing = self.spacing[self.nodules.patient_pos, :]
        self.nodules.origin = self.origin[self.nodules.patient_pos, :]
        self.nodules.img_size = self.shape[self.nodules.patient_pos, :]

    def _rescale_spacing(self):
        """Rescale spacing values and update nodules_info.

        This method should be called after any operation that
        changes shape of inner data.
        """
        if self.nodules is not None:
            self._refresh_nodules_info()
        return self

    @action
    @inbatch_parallel(init='_init_rebuild',
                      post='_post_rebuild', target='nogil')
    def resize(self, shape=(256, 256, 128), order=3, *args, **kwargs):    # pylint: disable=unused-argument, no-self-use
        """
        performs resize (change of shape) of each CT-scan in the batch.
            When called from Batch, changes Batch
            returns self
        args:
            shape: needed shape after resize in order x, y, z
                *note that the order of axes in data is z, y, x
                 that is, new patient shape = (shape[2], shape[1], shape[0])
            n_workers: number of threads used (degree of parallelism)
                *note: available in the result of decoration of the function
                above
            order: the order of interpolation (<= 5)
                large value improves precision, but slows down the computaion
        example:
            shape = (256, 256, 128)
            Batch = Batch.resize(shape=shape, n_workers=20, order=2)
        """
        return resize_patient_numba

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):
        """Post-function for resize parallelization.

        gatherer of outputs from different workers for
            ops, requiring complete rebuild of batch._data
        args:
            new_batch: if True, returns new batch with data
                agregated from workers_ouputs
        """
        # TODO: process errors
        batch = super()._post_rebuild(all_outputs, new_batch, **kwargs)
        batch.nodules = self.nodules
        batch._rescale_spacing()
        if self.mask is not None:
            batch.create_mask()
        return batch

    def make_xip(self, step=2, depth=10, func='max',
                 projection='axial', *args, **kwargs):    # pylint: disable=unused-argument, no-self-use
        logger.warning("There is no implementation of make_xip method for " +
                       "CTImagesMaskedBatch. Nothing happened.")
        return self

    def flip(self):
        logger.warning("There is no implementation of flip method for class " +
                       "CTIMagesMaskedBatch. Nothing happened")
        return self
