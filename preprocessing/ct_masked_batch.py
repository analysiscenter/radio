# pylint: disable=no-member
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
"""Contains class CTImagesMaskedBatch for storing masked Ct-scans."""
import logging

import numpy as np
from numba import njit
from skimage import measure
from .ct_batch import CTImagesBatch
from .mask import make_mask_numba
from .histo import sample_histo3d
from .crop import make_central_crop
from ..dataset import action, any_action_failed, DatasetIndex

LOGGING_FMT = (u"%(filename)s[LINE:%(lineno)d]#" +
               "%(levelname)-8s [%(asctime)s]  %(message)s")
logging.basicConfig(format=LOGGING_FMT, level=logging.DEBUG)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@njit(nogil=True)
def get_nodules_numba(data, positions, size):
    """ Fetch nodules from array by array of starting positions.

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
    """ Class for storing masked batch of ct-scans.

    Allows to load info about cancer nodules, then create cancer-masks
        for each patient. Created masks are stored in self.masks

    New attrs:
        1. masks: ndarray of masks
        2. nodules: record array with info about cancer nodules

    Important methods:
        1. fetch_nodules_info(self, nodules_df)
            function for loading info about nodules from annotations-df

        2. create_mask()
            given that self.nodules is filled (e.g. by calling fetch_nodules_info),
            this method fills self.masks-attribute with cancer-masks

        3. resize(self, shape=(128, 256, 256), order=3)
            transform shape of all scans in batch to supplied shape
            if masks are loaded, they are are also resized

        *Note: spacing, origin are recalculated when resize is executed
            As a result, load_mask can be also executed after resize
    """
    # record array contains the following information about nodules:
    # - self.nodules.nodule_center -- ndarray(num_nodules, 3) centers of
    #   nodules in world coords;
    # - self.nodules.nodule_size -- ndarray(num_nodules, 3) sizes of
    #   nodules along z, y, x in world coord;
    # - self.nodules.img_size -- ndarray(num_nodules, 3) sizes of images of
    #   patient data corresponding to nodules;
    # - self.nodules.offset -- ndarray(num_nodules, 3) of biases of
    #   patients which correspond to nodules;
    # - self.nodules.spacing -- ndarray(num_nodules, 3) of spacinf attribute
    #   of patients which correspond to nodules;
    # - self.nodules.origin -- ndarray(num_nodules, 3) of origin attribute
    #   of patients which correspond to nodules;
    nodules_dtype = np.dtype([('patient_pos', np.int, 1),
                              ('offset', np.int, (3,)),
                              ('img_size', np.int, (3,)),
                              ('nodule_center', np.float, (3,)),
                              ('nodule_size', np.float, (3,)),
                              ('spacing', np.float, (3,)),
                              ('origin', np.float, (3,))])

    @staticmethod
    def make_indices(size):
        """ Generate list of batch indices of given size.

        Take number of indices as input parameter size and
        generates list of random indices of length size.

        Args:
        - size: size of list with indices;
        """
        return [CTImagesMaskedBatch.make_filename() for i in range(size)]

    def __init__(self, index, *args, **kwargs):
        """ Initialization of CTImagesMaskedBatch.

        Initialize CTImagesMaskedBatch with index.
        """
        super().__init__(index, *args, **kwargs)
        self.masks = None
        self.nodules = None
        self.labels = None

    @property
    def components(self):
        """ Components-property.

        See doc of base batch from dataset for information.
        In short, these are names for components of tuple returned from __getitem__.
        """
        return 'images', 'masks', 'spacing', 'origin'

    @action
    def load(self, source=None, fmt='dicom', bounds=None,      # pylint: disable=arguments-differ
             origin=None, spacing=None, nodules=None, masks=None,
             src_blosc=None):
        """ Load data in masked batch of patients.

        Args:
        - source: source array with skyscraper, needed if fmt is 'ndarray';
        - fmt: type of source data; possible values are 'raw' and 'ndarray';
        - src_blosc: iterable with components of batch that should be loaded from blosc.
                Needed only if fmt='blosc'. If None, all components are loaded;
        Returns:
        - self;

        Examples:
        >>> index = FilesIndex(path="/some/path/*.mhd, no_ext=True")
        >>> batch = CTImagesMaskedBatch(index)
        >>> batch.load(fmt='raw')

        >>> batch.load(src=source_array, fmt='ndarray', bounds=bounds,
        ...            origin=origin_dict, spacing=spacing_dict)
        """
        params = dict(source=source, bounds=bounds, origin=origin, spacing=spacing)
        if fmt == 'ndarray':
            self._init_data(**params)
            self.nodules = nodules
            self.masks = masks
        else:
            # TODO check this
            super().load(fmt=fmt, **params, src_blosc=src_blosc)
        return self

    @action
    def dump(self, dst, src=None, fmt='blosc'):                # pylint: disable=arguments-differ
        """ Dump scans data (3d-array) on specified path in specified format

        Args:
            dst: general folder in which all patients' data should be put
            src: component(s) that we need to dump (smth iterable or string). If not
                supplied, dump all components + shapes of scans
            fmt: format of dump. Currently only blosc-format is supported;
                in this case folder for each patient is created, patient's data
                is put into images.blk, attributes are put into files attr_name.cpkl
                (e.g., spacing.cpkl)

        See docstring of parent-batch for examples.
        """
        # if src is not supplied, dump all components and shapes
        if src is None:
            src = self.components + ('images_shape', )

        # convert src to iterable 1d-array
        src = np.asarray(src).reshape(-1)

        if 'masks' in src and 'images_shape' not in src:
            src = tuple(src) + ('images_shape', )

        # execute parent-method
        super().dump(dst=dst, src=src, fmt=fmt)

        return self

    def get_pos(self, data, component, index):
        """ Return a posiiton of a component in data for a given index

        *NOTE: this is an overload of get_pos from base Batch-class,
            see corresponding docstring for detailed explanation.
        """
        if data is None:
            ind_pos = self._get_verified_pos(index)
            if component in ['images', 'masks']:
                return slice(self.lower_bounds[ind_pos], self.upper_bounds[ind_pos])
            else:
                return slice(ind_pos, ind_pos + 1)
        else:
            return index

    @property
    def num_nodules(self):
        """ Get number of nodules in CTImagesMaskedBatch.

        This property returns the number
        of nodules in CTImagesMaskedBatch. If fetch_nodules_info
        method has not been called yet returns -1.
        """
        if self.nodules is not None:
            return self.nodules.patient_pos.shape[0]
        else:
            return 0

    @action
    def fetch_nodules_info(self, nodules_df, update=False, images_loaded=True):
        """ Extract nodules' info from nodules_df into attribute self.nodules.

        This method fetch info about all nodules in batch
        and put them in numpy record array which can be accessed outside
        the class by self.nodules. Record array self.nodules
        has 'spacing', 'origin', 'img_size' and 'bias' properties, each
        represented by ndarray(num_nodules, 3) referring to spacing, origin,
        image size and bound of patients which correspond to fetched nodules.
        Record array self.nodules also contains attributes 'center' and 'size'
        which contain information about center and size of nodules in
        world coordinate system, each of these properties is represented by
        ndarray(num_nodules, 3). Finally, self.nodules.patient_pos refers to
        positions of patients which correspond to stored nodules.
        Object self.nodules is used by some methods, for example, create mask
        or sample nodule batch, to perform transform from world coordinate
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

        num_nodules = nodules_df.shape[0]
        self.nodules = np.rec.array(np.zeros(num_nodules,
                                             dtype=self.nodules_dtype))
        counter = 0
        for pat_id, coordz, coordy, coordx, diam in nodules_df.itertuples():
            pat_pos = self.index.get_pos(pat_id)
            self.nodules.patient_pos[counter] = pat_pos
            self.nodules.nodule_center[counter, :] = np.array([coordz,
                                                               coordy,
                                                               coordx])
            self.nodules.nodule_size[counter, :] = np.array([diam, diam, diam])
            counter += 1

        self._refresh_nodules_info(images_loaded)
        return self

    @action
    def fetch_nodules_from_mask(self, images_loaded=True):
        """ Fetch nodules info(centers and sizes) from masks.

        First of all, runs skimage.measure.labels for fetching nodules regions
        from mask. After that extracts nodules info from segmented regions
        and put this information in self.nodules record array.

        Args:
        - images_loaded: bool, whether 'images' component is loaded or not;
        this argument is required by self._refresh_nodules_info method;

        Returns:
        - self, batch;
        """
        nodules_list = []
        for pos in range(len(self)):
            mask = self.get(pos, 'masks')
            mask_labels = measure.label(mask, background=0)
            for props in measure.regionprops(np.int16(mask_labels)):
                center = np.asarray((props.centroid[0],
                                     props.centroid[1],
                                     props.centroid[2]), dtype=np.float)
                center = center * self.spacing[pos] + self.origin[pos]

                diameter = np.asarray([props.equivalent_diameter / 2] * 3, dtype=np.float)
                diameter = diameter * self.spacing[pos]
                nodules_list.append({'patient_pos': pos,
                                     'nodule_center': center,
                                     'nodule_size': diameter})

        num_nodules = len(nodules_list)
        self.nodules = np.rec.array(np.zeros(num_nodules, dtype=self.nodules_dtype))
        for i, nodule in enumerate(nodules_list):
            self.nodules.patient_pos[i] = nodule['patient_pos']
            self.nodules.nodule_center[i, :] = nodule['nodule_center']
            self.nodules.nodule_size[i, :] = nodule['nodule_size']
        self._refresh_nodules_info(images_loaded)
        return self

    # TODO think about another name of method
    def _fit_into_bounds(self, size, variance=None):
        """ Fetch start pixel coordinates of all nodules.

        This method returns start pixel coordinates of all nodules
        in batch. Note that all nodules are considered to have the
        fixed size defined by argument size: if nodule is out of
        patient's 3d image bounds than it's center is shifted.

        Args:
        - size: list, tuple of numpy array of length 3 with pixel
        size of nodules;
        - covariance: ndarray(3, ) diagonal elements
        of multivariate normal distribution used for sampling random shifts
        along [z, y, x] correspondingly;
        """
        size = np.array(size, dtype=np.int)

        center_pix = np.abs(self.nodules.nodule_center -
                            self.nodules.origin) / self.nodules.spacing
        start_pix = (np.rint(center_pix) - np.rint(size / 2))
        if variance is not None:
            start_pix += np.random.multivariate_normal(np.zeros(3),
                                                       np.diag(variance),
                                                       self.nodules.patient_pos.shape[0])
        end_pix = start_pix + size

        bias_upper = np.maximum(end_pix - self.nodules.img_size, 0)
        start_pix -= bias_upper
        end_pix -= bias_upper

        bias_lower = np.maximum(-start_pix, 0)
        start_pix += bias_lower
        end_pix += bias_lower

        return (start_pix + self.nodules.offset).astype(np.int)

    @action
    def create_mask(self):
        """ Load mask data for using nodule's info.

        Load mask into self.masks using info in attribute self.nodules_info.
        *Note: nodules info must be loaded before the call of this method.
        """
        if self.nodules is None:
            logger.warning("Info about nodules location must " +
                           "be loaded before calling this method. " +
                           "Nothing happened.")
        self.masks = np.zeros_like(self.images)

        center_pix = np.abs(self.nodules.nodule_center -
                            self.nodules.origin) / self.nodules.spacing
        start_pix = (center_pix - np.rint(self.nodules.nodule_size /
                                          self.nodules.spacing / 2))
        start_pix = np.rint(start_pix).astype(np.int)
        make_mask_numba(self.masks, self.nodules.offset,
                        self.nodules.img_size + self.nodules.offset, start_pix,
                        np.rint(self.nodules.nodule_size / self.nodules.spacing))

        return self

    def fetch_mask(self, shape):
        """ Create scaled mask using nodule info from self

        Args:
            shape: requiring shape of mask to be created

        Return:
            3d-array with mask in form of skyscraper

        # TODO: one part of code from here repeats create_mask function
            better to unify these two func
        """
        if self.nodules is None:
            logger.warning("Info about nodules location must " +
                           "be loaded before calling this method. " +
                           "Nothing happened.")
        mask = np.zeros(shape=(len(self) * shape[0], ) + tuple(shape[1:]))

        # infer scale factor; assume patients are already resized to equal shapes
        scale_factor = np.asarray(shape) / self.images_shape[0, :]

        # get rescaled nodule-centers, nodule-sizes, offsets, locs of nod starts
        center_scaled = (np.abs(self.nodules.nodule_center - self.nodules.origin) /
                         self.nodules.spacing * scale_factor)
        start_scaled = (center_scaled - scale_factor * self.nodules.nodule_size /
                        self.nodules.spacing / 2)
        start_scaled = np.rint(start_scaled).astype(np.int)
        offset_scaled = np.rint(self.nodules.offset * scale_factor).astype(np.int)
        img_size_scaled = np.rint(self.nodules.img_size * scale_factor).astype(np.int)
        nod_size_scaled = (np.rint(scale_factor * self.nodules.nodule_size /
                                   self.nodules.spacing)).astype(np.int)
        # put nodules into mask
        make_mask_numba(mask, offset_scaled, img_size_scaled + offset_scaled,
                        start_scaled, nod_size_scaled)
        # return ndarray-mask
        return mask

    # TODO rename function to sample_random_nodules_positions
    def sample_random_nodules(self, num_nodules, nodule_size, histo=None):
        """ Sample random nodules from CTImagesBatchMasked skyscraper.

        Samples random num_nodules' lower_bounds coordinates
        and stack obtained data into ndarray(l, 3) then returns it.
        First dimension of that array is just an index of sampled
        nodules while second points out pixels of start of nodules
        in BatchCt skyscraper. Each nodule have shape
        defined by parameter size. If size of patients' data along
        z-axis is not the same for different patients than
        NotImplementedError will be raised.

        Args:
        - num_nodules: number of random nodules to sample from BatchCt data;
        - nodule_size: ndarray(3, ) nodule size in number of pixels;
        - histo: 3d-histogram, represented by tuple (bins, edges) - format of
            np.histogramdd;

        return
        - ndarray(l, 3) of int that contains information
        about starting positions
        of sampled nodules in BatchCt skyscraper along each axis.
        First dimension is used to index nodules
        while the second one refers to various axes.

        *Note: [zyx]-ordering is used;
        """
        all_indices = np.arange(len(self))
        sampled_indices = np.random.choice(all_indices, num_nodules, replace=True)

        offset = np.zeros((num_nodules, 3))
        offset[:, 0] = self.lower_bounds[sampled_indices]
        data_shape = self.images_shape[sampled_indices, :]

        # if supplied, use histogram as the sampler
        if histo is None:
            sampler = lambda size: np.random.rand(size, 3)
        else:
            sampler = lambda size: sample_histo3d(histo, size)

        samples = sampler(size=num_nodules) * (data_shape - nodule_size)

        if histo is not None:
            samples /= data_shape

        return np.asarray(samples + offset, dtype=np.int)

    @action
    def sample_nodules(self, nodule_size, all_cancerous=False, batch_size=None, share=0.8,
                       variance=None, mask_shape=None, histo=None):
        """ Fetch random cancer and non-cancer nodules from batch.

        Fetch nodules from CTImagesBatchMasked into ndarray(l, m, k).

        Args:
        - nodules_df: dataframe of csv file with information
            about nodules location;
        - all_cancerous: if True, resulting batch will contain only cancerous
            nodules. Arg batch_size in this case is not needed.
        - batch_size: number of nodules in the output batch. Must be supplied
            whenever all_cancerous=False.
        - nodule_size: size of nodule along axes.
            Must be list, tuple or ndarray(3, ) of integer type;
            (Note: using zyx ordering)
        - share: share of cancer nodules in the batch.
            If source CTImagesBatch contains less cancer
            nodules than needed random nodules will be taken;
        - variance: variances of normally distributed random shifts of
            nodules' first pixels
        - mask_shape: needed shape of mask in (z, y, x)-order. If not None,
            masks of nodules will be scaled to shape=mask_shape
        - histo: 3d-histogram in np-histogram format, that is used for sampling
            non-cancerous crops

        Return:
            A batch with cancerous and non-cancerous nodules in a proportion defined by
                share with total batch_size nodules. If all_cancerous set to True, args
                share and batch_size are ignored and resulting batch consists of all
                cancerous nodules stored in batch.
        """
        # make sure that nodules' info is fetched and args are OK
        if self.nodules is None:
            raise AttributeError("Info about nodules location must " +
                                 "be loaded before calling this method")
        if variance is not None:
            variance = np.asarray(variance, dtype=np.int)
            variance = variance.flatten()
            if len(variance) != 3:
                logger.warning('Argument variance be np.array-like' +
                               'and has shape (3,). ' +
                               'Would be used no-scale-shift.')
                variance = None

        if not all_cancerous and batch_size is None:
            raise ValueError('Either supply batch_size or set all_cancerous to True')

        # choose cancerous nodules' starting positions
        nodule_size = np.asarray(nodule_size, dtype=np.int)
        batch_size = self.num_nodules if all_cancerous else batch_size
        cancer_n = int(share * batch_size) if not all_cancerous else self.num_nodules
        cancer_n = self.num_nodules if cancer_n > self.num_nodules else cancer_n
        if self.num_nodules == 0:
            cancer_nodules = np.zeros((0, 3))
        else:
            # adjust cancer nodules' starting positions s.t. nodules fit into scan-boxes
            cancer_nodules = self._fit_into_bounds(nodule_size, variance=variance)

            # randomly select needed number of cancer nodules (their starting positions)
            sample_indices = np.random.choice(np.arange(self.num_nodules),
                                              size=cancer_n, replace=False)
            cancer_nodules = cancer_nodules[sample_indices, :]

        nodules_st_pos = cancer_nodules

        # if non-cancerous nodules are needed, add random starting pos
        if batch_size - cancer_n > 0:
            # sample starting positions for (most-likely) non-cancerous crops
            random_nodules = self.sample_random_nodules(batch_size - cancer_n,
                                                        nodule_size, histo=histo)

            # concat non-cancerous and cancerous crops' starting positions
            nodules_st_pos = np.vstack([nodules_st_pos, random_nodules]).astype(np.int)  # pylint: disable=no-member

        # obtain nodules' scans by cropping from self.images
        images = get_nodules_numba(self.images, nodules_st_pos, nodule_size)

        # if mask_shape not None, compute scaled mask for the whole batch
        # scale also nodules' starting positions and nodules' shapes
        if mask_shape is not None:
            scale_factor = np.asarray(mask_shape) / np.asarray(nodule_size)
            batch_mask_shape = np.rint(scale_factor * self.images_shape[0, :]).astype(np.int)
            batch_mask = self.fetch_mask(batch_mask_shape)
            nodules_st_pos = np.rint(scale_factor * nodules_st_pos).astype(np.int)
        else:
            batch_mask = self.masks
            mask_shape = nodule_size

        # crop nodules' masks
        masks = get_nodules_numba(batch_mask, nodules_st_pos, mask_shape)

        # build noudles' batch
        bounds = np.arange(batch_size + 1) * nodule_size[0]
        ds_index = DatasetIndex(self.make_indices(batch_size))
        nodules_batch = type(self)(ds_index)
        nodules_batch.load(source=images, fmt='ndarray', bounds=bounds)

        # TODO add info about nodules by changing self.nodules
        nodules_batch.masks = masks
        return nodules_batch

    @action
    def update_nodules_histo(self, histo):
        """ Update histogram of nodules' locations in pixel coords using info
                about cancer nodules from batch.

        Args:
            histo: list of len=2, where the first item contains histogram bins: array of
                shape=(number of bins X number of bins) with number of elems in bins, while the
                second item contains edges of histogram bins:
                list of 3 1darrays of shape=(number of bins + 1).
                NOTE: this is almost np.histogram format. The only difference is that np.histogram
                returns tuple of len=2, not list.

        Return:
            self

        NOTE: execute this action only after fetch_nodules_info.
        """
        # infer bins' bounds from histo
        bins = histo[1]

        # get cancer_nodules' centers in pixel coords
        center_pix = np.abs(self.nodules.nodule_center - self.nodules.origin) / self.nodules.spacing

        # update bins of histo
        histo_delta = np.histogramdd(center_pix, bins=bins)
        histo[0] += histo_delta[0]

        return self

    def get_axial_slice(self, patient_pos, height):
        """ Get tuple of slices (data slice, mask slice).

        Args:
            patient_pos: patient position in the batch
            height: height, take slices with number
                int(0.7 * number of slices for patient) from
                patient's scan and mask
        """
        margin = int(height * self.get(patient_pos, 'images').shape[0])
        if self.masks is not None:
            patch = (self.get(patient_pos, 'images')[margin, :, :],
                     self.get(patient_pos, 'masks')[margin, :, :])
        else:
            patch = (self.get(patient_pos, 'images')[margin, :, :], None)
        return patch

    def _refresh_nodules_info(self, images_loaded=True):
        """ Refresh self.nodules attributes [spacing, origin, img_size, bias].

        This method should be called when it is needed to make
        [spacing, origin, img_size, bias] attributes of self.nodules
        to correspond the structure of batch's inner data.

        Args:
            images_loaded: if set to True, assume that _bounds-attr is loaded
                and images are already loaded.
        """
        if images_loaded:
            self.nodules.offset[:, 0] = self.lower_bounds[self.nodules.patient_pos]
            self.nodules.img_size = self.images_shape[self.nodules.patient_pos, :]

        self.nodules.spacing = self.spacing[self.nodules.patient_pos, :]
        self.nodules.origin = self.origin[self.nodules.patient_pos, :]

    def _rescale_spacing(self):
        """ Rescale spacing values and update nodules_info.

        This method should be called after any operation that
        changes shape of inner data.
        """
        if self.nodules is not None:
            self._refresh_nodules_info()
        return self

    def _post_mask(self, list_of_arrs, **kwargs):
        """ Concatenate outputs of different workers and put the result in mask-attr

        Args:
            list_of_arrs: list of ndarays, with each ndarray representing a
                patient's mask
        """
        if any_action_failed(list_of_arrs):
            raise ValueError("Failed while parallelizing")

        new_masks = np.concatenate(list_of_arrs, axis=0)
        self.masks = new_masks

        return self

    def _init_load_blosc(self, **kwargs):
        """ Init-func for load from blosc. Fills images/masks-components with zeroes
                if the components are to be updated.

        Args:
            src: iterable of components that need to be loaded
        Return
            list of ids of batch-items
        """
        # read shapes, fill the components with zeroes if masks, images need to be updated
        if 'masks' in kwargs['src'] or 'images' in kwargs['src']:
            slice_shape = self._preload_shapes()
            skysc_shape = (self._bounds[-1], ) + slice_shape

            # fill needed comps with zeroes
            for source in {'images', 'masks'} & set(kwargs['src']):
                setattr(self, source, np.zeros(skysc_shape))

        return self.indices

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):
        """ Post-function for resize parallelization.

        gatherer of outputs from different workers for
            ops, requiring complete rebuild of batch._data
        args:
            new_batch: if True, returns new batch with data
                agregated from workers_ouputs
        """
        # TODO: process errors
        batch = super()._post_rebuild(all_outputs, new_batch, **kwargs)
        batch.nodules = self.nodules
        batch._rescale_spacing()  # pylint: disable=protected-access
        if self.masks is not None:
            batch.create_mask()
        return batch

    @action
    def make_xip(self, step=2, depth=10, func='max',
                 projection='axial', *args, **kwargs):
        """ Compute xip of source CTImage along given x with given step and depth.

        Call parent variant of make_xip then change nodules sizes'
        via calling _update_nodule_size and create new mask that corresponds
        to data after transform.
        """
        batch = super().make_xip(step=step, depth=depth, func=func,
                                 projection=projection, *args, **kwargs)

        batch.nodules = self.nodules
        if projection == 'axial':
            _projection = 0
        elif projection == 'coronal':
            _projection = 1
        elif projection == 'sagital':
            _projection = 2
        batch.nodules.nodule_size[:, _projection] += (depth
                                                      * self.nodules.spacing[:, _projection])  # pylint: disable=unsubscriptable-object
        batch.spacing = self.rescale(batch[0].shape)
        batch._rescale_spacing()   # pylint: disable=protected-access
        if self.masks is not None:
            batch.create_mask()
        return batch

    @action
    def central_crop(self, crop_size, crop_mask=False, **kwargs):
        """ Make crop with given size from center of images.

        Args:
        - crop_size: tuple(int, int, int), size of crop;

        NOTE: this method should be rewritten with the use of
        inheritance;
        """
        crop_size = np.asarray(crop_size)
        img_shapes = [np.asarray(self.get(i, 'images').shape) for i in range(len(self))]
        if any(np.any(shape < crop_size) for shape in img_shapes):
            raise ValueError("Crop size must be smaller than size of inner 3D images")

        cropped_images = []
        cropped_masks = []
        for i in range(len(self)):
            image = self.get(i, 'images')
            cropped_images.append(make_central_crop(image, crop_size))

            if crop_mask and self.masks is not None:
                mask = self.get(i, 'masks')
                cropped_masks.append(make_central_crop(mask, crop_size))

        self._bounds = np.cumsum([0] + [crop_size[0]] * len(self))
        self.images = np.concatenate(cropped_images, axis=0)
        if crop_mask and self.masks is not None:
            self.masks = np.concatenate(cropped_masks, axis=0)
        return self

    def flip(self):
        logger.warning("There is no implementation of flip method for class " +
                       "CTIMagesMaskedBatch. Nothing happened")
        return self

    @action
    def create_labels_by_mask(self, threshold):
        """ Create labels attribute that containes labels corresponding to masks.

        Args:
        - threshold: int, threshold for number of 1 pixel for considering
        item having cancer;

        Returns:
        - self, unchanged batch;
        """
        self.labels = np.asarray([np.sum(self.get(i, 'masks')) > threshold
                                  for i in range(len(self))], dtype=np.float)
        return self

    def unpack_data(self, y_component, dim_ordering='channels_last', **kwargs):
        """ Unpack data contained in batch for feeding in model.

        Args:
        - y_component: str, name of y_component to fetch, can be 'masks' or 'labels';
        - dim_ordering: str, can be 'channels_last' or 'channels_first';

        Returns:
        - x, y ndarrays;
        """
        x, y = [], [] if y_component is not None else None
        for i in range(len(self)):
            x.append(self.get(i, 'images'))
            if y_component == 'no_y':
                y.append(np.nan)
                continue
            elif y_component == 'masks':
                y.append(self.get(i, 'masks'))
            elif y_component == 'labels':
                y.append(self.labels[i])
        x, y = np.stack(x), np.stack(y)
        if dim_ordering == 'channels_last':
            x, y = x[..., np.newaxis], y[..., np.newaxis]
        elif dim_ordering == 'channels_first':
            x = x[:, np.newaxis, ...]
            if y_component == 'masks':
                y = y[:, np.newaxis, ...]
        return x, y

    @action
    def train_on_crop(self, model_name, y_component='labels',
                      dim_ordering='channels_last', **kwargs):
        """ Train model on crops of CT-scans contained in batch.

        Args:
        - model_name: str, name of classification model;
        - y_component: str, name of y component, can be 'masks' or 'labels';
        - dim_ordering: str, dimension ordering, can be 'channels_first' or 'channels_last';

        Returns:
        - self, unchanged CTImagesMaskedBatch;
        """
        _model = self.get_model_by_name(model_name)
        x, y_true = self.unpack_data(dim_ordering=dim_ordering,
                                     y_component=y_component, **kwargs)
        _model.train_on_batch(x, y_true)
        return self

    @action
    def predict_on_crop(self, model_name, dst_dict, y_component='labels',
                        dim_ordering='channels_last', **kwargs):
        """ Get predictions of model on crops of CT-scans contained in batch.

        This action-method get predictions of model specified by 'model_name'
        argument on crops contained in batch and extends 'dst_dict' with
        predictions. Keys of 'dst_dict' are indices and values are predictions.

        Args:
        - model_name: str, name of classification model;
        - dst_dict: dictionary that will be updated by predictions;

        - y_component: str, name of y component, can be 'masks' or 'labels';
        - dim_ordering: str, dimension ordering, can be 'channels_first' or 'channels_last';

        Returns:
        - self, unchanged CTImagesMaskedBatch;
        """
        _model = self.get_model_by_name(model_name)
        x, _ = self.unpack_data(dim_ordering=dim_ordering,
                                y_component=y_component, **kwargs)
        predicted_labels = _model.predict_on_batch(x)
        dst_dict.update(zip(self.indices, predicted_labels))
        return self

    @action
    def predict_on_scan(self, model_name, strides=(16, 32, 32), crop_shape=(32, 64, 64),
                        batch_size=4, y_component='labels', dim_ordering='channels_last'):
        """ Get predictions of the model on data contained in batch.

        Transforms scan data into patches of shape CROP_SHAPE and then feed
        this patches sequentially into model with name specified by
        argument 'model_name'; after that loads predicted masks or probabilities
        into 'masks' component of the current batch and returns it.

        Args:
        - model_name: str, name of model;
        - strides: tuple(int, int, int) strides for patching operation;
        - batch_size: int, number of patches to feed in model in one iteration;
        - y_component: str, name of y component, can be 'masks' or labels;
        - dim_ordering: str, dimension ordering, can be 'channels_first' or 'channels_last'

        Returns:
        - self, uncahnged CTImagesMaskedBatch;
        """
        _model = self.get_model_by_name(model_name)

        patches_arr = self.get_patches(patch_shape=crop_shape, stride=strides, padding='reflect')
        if dim_ordering == 'channels_first':
            patches_arr = patches_arr[:, np.newaxis, ...]
        elif dim_ordering == 'channels_last':
            patches_arr = patches_arr[..., np.newaxis]

        predictions = []
        for i in range(0, patches_arr.shape[0], batch_size):
            current_prediction = np.asarray(_model.predict_on_batch(patches_arr[i: i + batch_size, ...]))

            if y_component == 'labels':
                current_prediction = np.stack([np.ones(shape=(crop_shape)) * prob
                                               for prob in current_prediction.ravel()])

            current_prediction = np.squeeze(current_prediction)
            predictions.append(current_prediction)

        patches_mask = np.concatenate(predictions, axis=0)
        self.load_from_patches(patches_mask, stride=strides,
                               scan_shape=tuple(self.images_shape[0, :]),
                               data_attr='masks')
        return self
