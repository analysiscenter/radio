# pylint: disable=no-member
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

""" Batch class CTImagesMaskedBatch for storing CT-scans with masks. """

import logging

import numpy as np
import pandas as pd
from numba import njit
from skimage import measure

try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x

from .ct_batch import CTImagesBatch
from .mask import make_mask_numba, create_mask_reg
from .histo import sample_histo3d
from .crop import make_central_crop
from ..dataset import action, any_action_failed, DatasetIndex, SkipBatchException  # pylint: disable=no-name-in-module

LOGGING_FMT = (u"%(filename)s[LINE:%(lineno)d]#" +
               "%(levelname)-8s [%(asctime)s]  %(message)s")
logging.basicConfig(format=LOGGING_FMT, level=logging.DEBUG)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@njit(nogil=True)
def get_nodules_numba(data, positions, size):
    """ Fetch nodules from array by starting positions.

    Takes array with data of shape (z, y, x) from `batch`,
    ndarray(p, 3) with starting indices of nodules where p is number
    of nodules and size of type ndarray(3, ) which contains
    sizes of nodules along each axis. The output is 3d ndarray with nodules
    put in CTImagesBatch-compatible skyscraper structure.

    Parameters
    ----------
    data :       ndarray
                 CTImagesBatch `skyscraper` represented by 3D ndarray.
    positions :  ndarray(l, 3) of int
                 Contains nodules' starting indices along [zyx]-axis accordingly in `data`.
    size :       ndarray(3,) of int
                 Contains nodules' sizes along each axis (z,y,x).
    Note
    ----
    Dtypes of positions and size arrays must be the same.

    Returns
    -------
    ndarray
            3d ndarray with nodules
    """
    out_arr = np.zeros((np.int(positions.shape[0]), size[0], size[1], size[2]))

    n_positions = positions.shape[0]
    for i in range(n_positions):
        out_arr[i, :, :, :] = data[positions[i, 0]: positions[i, 0] + size[0],
                                   positions[i, 1]: positions[i, 1] + size[1],
                                   positions[i, 2]: positions[i, 2] + size[2]]

    return out_arr.reshape(n_positions * size[0], size[1], size[2])


class CTImagesMaskedBatch(CTImagesBatch):
    """ Batch class for storing batch of ct-scans with masks for nodules.

    Allows to load info about cancer nodules, then create cancer-masks
        for each patient. Created masks are stored in self.masks

    Parameters
    ----------
    index : dataset.index
            ids of scans to be put in a batch

    Attributes
    ----------
    components :  tuple of strings.
                  List names of data components of a batch, which are `images`,
                  `masks`, `origin` and `spacing`.
                  NOTE: Implementation of this property is required by Base class.
    num_nodules : int
                  number of nodules in batch
    images :      ndarray
                  contains ct-scans for all patients in batch.
    masks :       ndarray
                  contains masks for all patients in batch.
    nodules :     np.recarray
                  contains info on cancer nodules location.
                  record array contains the following information about nodules:
                   - self.nodules.nodule_center -- ndarray(num_nodules, 3) centers of
                     nodules in world coords;
                   - self.nodules.nodule_size -- ndarray(num_nodules, 3) sizes of
                     nodules along z, y, x in world coord;
                   - self.nodules.img_size -- ndarray(num_nodules, 3) sizes of images of
                     patient data corresponding to nodules;
                   - self.nodules.offset -- ndarray(num_nodules, 3) position of individual
                     patient scan inside batch;
                   - self.nodules.spacing -- ndarray(num_nodules, 3) of spacing attribute
                     of patients which correspond to nodules;
                   - self.nodules.origin -- ndarray(num_nodules, 3) of origin attribute
                     of patients which correspond to nodules.
    """

    nodules_dtype = np.dtype([('patient_pos', np.int, 1),
                              ('offset', np.int, (3,)),
                              ('img_size', np.int, (3,)),
                              ('nodule_center', np.float, (3,)),
                              ('nodule_size', np.float, (3,)),
                              ('spacing', np.float, (3,)),
                              ('origin', np.float, (3,))])

    @staticmethod
    def make_indices(size):
        """ Generate list of batch indices of given `size`.

        Parameters
        ----------
        size : int
               size of list with indices

        Returns
        -------
        list
            list of random indices
        """
        return np.array([CTImagesMaskedBatch.make_filename() for i in range(size)])

    def __init__(self, index, *args, **kwargs):
        """ Execute Batch construction and init of basic attributes

        Parameters
        ----------
        index : Dataset.Index class.
                Required indexing of objects (files).
        """
        super().__init__(index, *args, **kwargs)
        self.masks = None
        self.nodules = None
        self.inputs = None
        self.targets = None

    @property
    def components(self):
        """ Components' property.

        See doc of Base batch in dataset for information.

        Returns
        -------
        tuple
            names of components returned from __getitem__.
        """
        return 'images', 'masks', 'spacing', 'origin'

    @action
    def load(self, source=None, fmt='dicom', bounds=None,      # pylint: disable=arguments-differ
             origin=None, spacing=None, nodules=None, masks=None,
             src_blosc=None):
        """ Load data in batch with scans and masks.

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

        Examples
        --------

        >>> index = FilesIndex(path="/some/path/*.mhd, no_ext=True")
        >>> batch = CTImagesMaskedBatch(index)
        >>> batch.load(fmt='raw')

        >>> batch.load(src=source_array, fmt='ndarray', bounds=bounds,
        ...            origin=origin_dict, spacing=spacing_dict)
        """
        params = dict(source=source, bounds=bounds,
                      origin=origin, spacing=spacing)
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
        """ Dump scans to dst-folder in specified format.

        Parameters
        ----------
        dst : str
              destinatio-folder where all patients' data should be put
        src : str or list/tuple
              component(s) that we need to dump. If not
              supplied, dump all components + shapes of scans
        fmt : 'blosc'
              format of dump. Currently only blosc-format is supported;
              in this case folder for each patient is created, patient's data
              is put into images.blk, masks.blk,
              attributes are put into files attr_name.cpkl
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
        super().dump(dst=dst, src=src, fmt=fmt)  # pylint: disable=no-value-for-parameter

        return self

    def nodules_to_df(self, nodules):
        """ Convert nodules_info ndarray into pandas dataframe.

        Pandas DataFrame will contain following columns:
        'source_id' - id of source element of batch;
        'nodule_id' - generated id for nodules;
        'locZ', 'locY', 'locX' - coordinates of nodules' centers;
        'diamZ', 'diamY', 'diamX' - sizes of nodules along zyx axes;

        Parameters
        ----------
        nodules : ndarray of type nodules_info
            nodules_info type is defined inside of CTImagesMaskedBatch class.

        Returns
        -------
        pd.DataFrame
            centers, ids and sizes of nodules.
        """
        columns = ['nodule_id', 'source_id', 'locZ', 'locY',
                   'locX', 'diamZ', 'diamY', 'diamX']

        nodule_id = self.make_indices(nodules.shape[0])
        return pd.DataFrame({'source_id': self.indices[nodules.patient_pos],
                             'nodule_id': nodule_id,
                             'locZ': nodules.nodule_center[:, 0],
                             'locY': nodules.nodule_center[:, 1],
                             'locX': nodules.nodule_center[:, 2],
                             'diamZ': nodules.nodule_size[:, 0],
                             'diamY': nodules.nodule_size[:, 1],
                             'diamX': nodules.nodule_size[:, 2]}, columns=columns)

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
            if component in ['images', 'masks']:
                return slice(self.lower_bounds[ind_pos], self.upper_bounds[ind_pos])
            else:
                return slice(ind_pos, ind_pos + 1)
        else:
            return index

    @property
    def num_nodules(self):
        """ Get number of nodules in CTImagesMaskedBatch.

        Returns
        -------
        int
            number of nodules in CTImagesMaskedBatch.
            if fetch_nodules_info method has not been called yet returns 0.
        """
        if self.nodules is not None:
            return self.nodules.patient_pos.shape[0]
        else:
            return 0

    @action
    def fetch_nodules_info(self, nodules_df=None, nodules_records=None, update=False, images_loaded=True):
        """Extract nodules' info from nodules_df into attribute self.nodules.

        Parameters
        ----------
        nodules_df :      pd.DataFrame
                          contains:
                           - 'seriesuid': index of patient or series.
                           - 'z','y','x': coordinates of nodules center.
                           - 'diameter': diameter, in mm.
        nodules_records : np.recarray
                          if not None, should
                          contain the same fields as describe in Note.
        update :          bool
                          if False, warning appears to remind that nodules info
                          will be earased and recomputed.
        images_loaded :   bool
                          if True, i.e. `images` component is loaded,
                          and image_size is used to compute
                          correct nodules location inside `skyscraper`.
                          If False, it doesn't update info of location
                          inside `skyscraper`.


        Note
        ----
        The method fills in record array self.nodules that contains the following information about nodules:
                               - self.nodules.nodule_center -- ndarray(num_nodules, 3) centers of
                                 nodules in world coords;
                               - self.nodules.nodule_size -- ndarray(num_nodules, 3) sizes of
                                 nodules along z, y, x in world coord;
                               - self.nodules.img_size -- ndarray(num_nodules, 3) sizes of images of
                                 patient data corresponding to nodules;
                               - self.nodules.offset -- ndarray(num_nodules, 3) of biases of
                                 patients which correspond to nodules;
                               - self.nodules.spacing -- ndarray(num_nodules, 3) of spacinf attribute
                                 of patients which correspond to nodules;
                               - self.nodules.origin -- ndarray(num_nodules, 3) of origin attribute
                                 of patients which correspond to nodules.
                               - self.nodules.patient_pos -- ndarray(num_nodules, 1) refers to
                                 positions of patients which correspond to stored nodules.

        """
        if self.nodules is not None and not update:
            logger.warning("Nodules have already been extracted. " +
                           "Put update argument as True for refreshing")
            return self

        if nodules_records is not None:
            # load from record-array
            self.nodules = nodules_records

        else:
            # assume that nodules_df is supplied and load from it
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
        """ Fetch nodules info (centers and sizes) from masks.

        Runs skimage.measure.labels for fetching nodules regions
        from masks. Extracts nodules info from segmented regions
        and put this information in self.nodules np.recarray.

        Parameters
        ----------
        images_loaded : bool
                        if True, i.e. `images` component is loaded,
                        and image_size is used to compute
                        correct nodules location inside `skyscraper`.
                        If False, it doesn't update info of location
                        inside `skyscraper`.

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

                diameter = np.asarray(
                    [props.equivalent_diameter] * 3, dtype=np.float)
                diameter = diameter * self.spacing[pos]
                nodules_list.append({'patient_pos': pos,
                                     'nodule_center': center,
                                     'nodule_size': diameter})

        num_nodules = len(nodules_list)
        self.nodules = np.rec.array(
            np.zeros(num_nodules, dtype=self.nodules_dtype))
        for i, nodule in enumerate(nodules_list):
            self.nodules.patient_pos[i] = nodule['patient_pos']
            self.nodules.nodule_center[i, :] = nodule['nodule_center']
            self.nodules.nodule_size[i, :] = nodule['nodule_size']
        self._refresh_nodules_info(images_loaded)
        return self

    # TODO: another name of method
    def _fit_into_bounds(self, size, variance=None):
        """ Fetch start voxel coordinates of all nodules.

        Get start voxel coordinates of all nodules in batch.
        Note that all nodules are considered to have
        fixed same size defined by argument size: if nodule is out of
        patient's 3d image bounds than it's center is shifted to border.

        Parameters
        ----------
        size :     list or tuple of ndarrays
                   ndarray(3, ) with diameters of nodules in (z,y,x);
        variance : ndarray(3, )
                   diagonal elements of multivariate normal distribution,
                   for sampling random shifts along (z,y,x) correspondingly.

        Returns
        -------
        ndarray
                start coordinates (z,y,x) of all nodules in batch.
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
        """ Create `masks` component from `nodules` component.

        Note
        ----
        `nodules` must be not None before calling this method.
        see fetch_nodules_info() for more details.
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
        """ Create `masks` component of different size then `images`,
        using `nodules` component.

        Parameters
        ----------
        shape : tuple, list or ndarray of int.
                (z_dim,y_dim,x_dim), shape of mask to be created.

        Returns
        -------
        ndarray
               3d array with masks in form of `skyscraper`

        # TODO: one part of code from here repeats create_mask function
            better to unify these two func
        """
        if self.nodules is None:
            logger.warning("Info about nodules location must " +
                           "be loaded before calling this method. " +
                           "Nothing happened.")
        mask = np.zeros(shape=(len(self) * shape[0], ) + tuple(shape[1:]))

        # infer scale factor; assume patients are already resized to equal
        # shapes
        scale_factor = np.asarray(shape) / self.images_shape[0, :]

        # get rescaled nodule-centers, nodule-sizes, offsets, locs of nod
        # starts
        center_scaled = (np.abs(self.nodules.nodule_center - self.nodules.origin) /
                         self.nodules.spacing * scale_factor)
        start_scaled = (center_scaled - scale_factor * self.nodules.nodule_size /
                        self.nodules.spacing / 2)
        start_scaled = np.rint(start_scaled).astype(np.int)
        offset_scaled = np.rint(self.nodules.offset *
                                scale_factor).astype(np.int)
        img_size_scaled = np.rint(
            self.nodules.img_size * scale_factor).astype(np.int)
        nod_size_scaled = (np.rint(scale_factor * self.nodules.nodule_size /
                                   self.nodules.spacing)).astype(np.int)
        # put nodules into mask
        make_mask_numba(mask, offset_scaled, img_size_scaled + offset_scaled,
                        start_scaled, nod_size_scaled)
        # return ndarray-mask
        return mask

    # TODO rename function to sample_random_nodules_positions
    def sample_random_nodules(self, num_nodules, nodule_size, histo=None):
        """ Sample random nodules positions in CTImagesBatchMasked.

        Samples random nodules positions in ndarray. Each nodule have shape
        defined by `nodule_size`. If size of patients' data along z-axis
        is not the same for different patients, NotImplementedError will be raised.

        Parameters
        ----------
        num_nodules : int
                      number of nodules to sample from dataset.
        nodule_size : ndarray(3, )
                      crop shape along (z,y,x).
        histo :       tuple
                      np.histogram()'s output.
                      3d-histogram, represented by tuple (bins, edges)

        Returns
        -------
        ndarray
                ndarray(num_nodules, 3). 1st array's dim is an index of sampled
                nodules, 2nd points out start positions (integers) of nodules
                in batch `skyscraper`.
        """
        all_indices = np.arange(len(self))
        sampled_indices = np.random.choice(
            all_indices, num_nodules, replace=True)

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

        return np.asarray(samples + offset, dtype=np.int), sampled_indices

    @action
    def sample_nodules(self, batch_size, nodule_size=(32, 64, 64), share=0.8, variance=None,        # pylint: disable=too-many-locals, too-many-statements
                       mask_shape=None, histo=None):
        """ Sample random crops of `images` and `masks` from batch.


        Create random crops, both with and without nodules in it, from input batch.

        Parameters
        ----------
        batch_size :    int
                        number of nodules in the output batch. Required,
                        if share=0.0. If None, resulting batch will include all
                        cancerous nodules
        nodule_size :   tuple, list or ndarray of int
                        crop shape along (z,y,x).
        share :         float
                        share of cancer crops in the batch.
                        if input CTImagesBatch contains less cancer
                        nodules than needed random nodules will be taken;
        variance :      tuple, list of float
                        variances of normally distributed random shifts of
                        nodules' start positions
        mask_shape :    tuple, list or ndarray
                        size of `masks` crop in (z, y, x)-order. If not None,
                        crops with masks would be of mask_shape.
                        If None, mask crop shape would be equal to crop_size.
        histo :         tuple
                        np.histogram()'s output.
                        Used for sampling non-cancerous crops

        Returns
        -------
        Batch
                batch with cancerous and non-cancerous crops in a proportion defined by
                `share` with total `batch_size` nodules. If `share` == 1.0, `batch_size`
                is None, resulting batch consists of all cancerous crops stored in batch.
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

        if share == 0.0 and batch_size is None:
            raise ValueError('Either supply batch_size or set share to positive number')

        # pos of batch-items that correspond to crops
        crops_indices = np.zeros(0, dtype=np.int16)

        # infer the number of cancerous nodules and the size of batch
        batch_size = batch_size if batch_size is not None else 1.0 / share * self.num_nodules
        cancer_n = int(share * batch_size)
        batch_size = int(batch_size)
        cancer_n = self.num_nodules if cancer_n > self.num_nodules else cancer_n

        if batch_size == 0:
            raise SkipBatchException('Batch of zero size cannot be passed further through the workflow')

        # choose cancerous nodules' starting positions
        nodule_size = np.asarray(nodule_size, dtype=np.int)
        if self.num_nodules == 0:
            cancer_nodules = np.zeros((0, 3))
        else:
            # adjust cancer nodules' starting positions s.t. nodules fit into
            # scan-boxes
            cancer_nodules = self._fit_into_bounds(
                nodule_size, variance=variance)

            # randomly select needed number of cancer nodules (their starting
            # positions)
            sample_indices = np.random.choice(np.arange(self.num_nodules),
                                              size=cancer_n, replace=False)
            cancer_nodules = cancer_nodules[sample_indices, :]

            # store scans-indices for chosen crops
            cancerous_indices = self.nodules.patient_pos[sample_indices].reshape(-1)
            crops_indices = np.concatenate([crops_indices, cancerous_indices])

        nodules_st_pos = cancer_nodules

        # if non-cancerous nodules are needed, add random starting pos
        if batch_size - cancer_n > 0:
            # sample starting positions for (most-likely) non-cancerous crops
            random_nodules, random_indices = self.sample_random_nodules(batch_size - cancer_n,
                                                                        nodule_size, histo=histo)

            # concat non-cancerous and cancerous crops' starting positions
            nodules_st_pos = np.vstack([nodules_st_pos, random_nodules]).astype(
                np.int)  # pylint: disable=no-member

            # store scan-indices for randomly chose crops
            crops_indices = np.concatenate([crops_indices, random_indices])

        # obtain nodules' scans by cropping from self.images
        images = get_nodules_numba(self.images, nodules_st_pos, nodule_size)

        # if mask_shape not None, compute scaled mask for the whole batch
        # scale also nodules' starting positions and nodules' shapes
        if mask_shape is not None:
            scale_factor = np.asarray(mask_shape) / np.asarray(nodule_size)
            batch_mask_shape = np.rint(
                scale_factor * self.images_shape[0, :]).astype(np.int)
            batch_mask = self.fetch_mask(batch_mask_shape)
            nodules_st_pos = np.rint(
                scale_factor * nodules_st_pos).astype(np.int)
        else:
            batch_mask = self.masks
            mask_shape = nodule_size

        # crop nodules' masks
        masks = get_nodules_numba(batch_mask, nodules_st_pos, mask_shape)

        # build nodules' batch
        bounds = np.arange(batch_size + 1) * nodule_size[0]
        crops_spacing = self.spacing[crops_indices]
        offset = np.zeros((batch_size, 3))
        offset[:, 0] = self.lower_bounds[crops_indices]
        crops_origin = self.origin[crops_indices] + crops_spacing * (nodules_st_pos - offset)
        names_gen = zip(self.indices[crops_indices], self.make_indices(batch_size))
        ix_batch = ['_'.join([prefix, random_str]) for prefix, random_str in names_gen]
        nodules_batch = type(self)(DatasetIndex(ix_batch))
        nodules_batch.load(source=images, fmt='ndarray', bounds=bounds, spacing=crops_spacing, origin=crops_origin)

        # set masks
        nodules_batch.masks = masks

        # set nodules info in nodules' batch
        nodules_records = [self.nodules[self.nodules.patient_pos == crop_pos] for crop_pos in crops_indices]
        new_patient_pos = []
        for i, records in enumerate(nodules_records):
            new_patient_pos += [i] * len(records)
        new_patient_pos = np.array(new_patient_pos)
        nodules_records = np.concatenate(nodules_records)
        nodules_records = nodules_records.view(np.recarray)
        nodules_records.patient_pos = new_patient_pos
        nodules_batch.fetch_nodules_info(nodules_records=nodules_records)

        # leave out nodules with zero-intersection with crops' boxes
        nodules_batch._filter_nodules_info()                                                     # pylint: disable=protected-access

        return nodules_batch

    @action
    def sample_dump(self, dst, n_iters, nodule_size=(32, 64, 64), batch_size=20, share=0.8, **kwargs):
        """ Perform sample_nodules and dump on the same batch n_iters times.

        Can be used for fast creation of large datasets of cancerous/non-cancerous crops.

        Parameters
        ----------
        dst :         str
                      folder to dump nodules in.
        n_iters :     int
                      number of iterations to be performed.
        nodule_size : tuple or list
                      shape of sampled nodules.
        batch_size :  int or None
                      size of generated batches.
        share :       float
                      share of cancer nodules. See docstring of sample_nodules for more info
                      about possible combinations of parameters share and batch_size.
        **kwargs :    dict
                      additional arguments supplied into sample_nodules. See docstring
                      of sample_nodules for more info.
        """
        for _ in range(n_iters):
            nodules = self.sample_nodules(batch_size=batch_size, nodule_size=nodule_size, share=share, **kwargs)
            nodules = nodules.dump(dst=dst)

        return self

    @action
    def update_nodules_histo(self, histo):
        """ Update histogram of nodules' locations using nodules locations from batch.

        Parameters
        ----------
        histo : list
                list(np.histogram()), used for sampling cancerous locations.


        Note
        ----
        Execute action only after .fetch_nodules_info().
        """
        # infer bins' bounds from histo
        bins = histo[1]

        # get cancer_nodules' centers in voxel coords
        center_pix = np.abs(self.nodules.nodule_center -
                            self.nodules.origin) / self.nodules.spacing

        # update bins of histo
        histo_delta = np.histogramdd(center_pix, bins=bins)
        histo[0] += histo_delta[0]

        return self

    def get_axial_slice(self, patient_pos, height):
        """ Get tuple of `images` slice and `masks` slice by patient and slice position.

        Parameters
        ----------
        patient_pos : int
                      patient position in the batch
        height :      float
                      number of slice (z-axis), scaled to [0:1]
                      used to get slice with position:
                      int(height * number_of slices_for_patient) from
                      patient's scan and mask.

        Returns
        -------
        tuple
             (images_slice,masks_slice) by patient_pos and number of slice
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

        This method is called to update [spacing, origin, img_size, bias]
        attributes of self.nodules because batch's inner data has changed,
        e.g. after resize.

        Parameters
        ----------
        images_loaded : bool
                        if True, assumes that `_bounds` attribute is computed,
                        i.e. either `masks` and/or `images` are loaded.
        """
        if images_loaded:
            self.nodules.offset[:, 0] = self.lower_bounds[
                self.nodules.patient_pos]
            self.nodules.img_size = self.images_shape[
                self.nodules.patient_pos, :]

        self.nodules.spacing = self.spacing[self.nodules.patient_pos, :]
        self.nodules.origin = self.origin[self.nodules.patient_pos, :]

    def _filter_nodules_info(self):
        """ Filter record-array self.nodules s.t. only records about cancerous nodules
        that have non-zero intersection with scan-boxes be present.

        Note
        ----
        can be called only after execution of fetch_nodules_info and _refresh_nodules_info
        """
        # nodules start and trailing pixel-coords
        center_pix = (self.nodules.nodule_center - self.nodules.origin) / self.nodules.spacing
        start_pix = center_pix - np.rint(self.nodules.nodule_size / self.nodules.spacing / 2)
        start_pix = np.rint(start_pix).astype(np.int)
        end_pix = start_pix + np.rint(self.nodules.nodule_size / self.nodules.spacing)

        # find nodules with no intersection with scan-boxes
        nods_images_shape = self.images_shape[self.nodules.patient_pos]
        start_mask = np.any(start_pix >= nods_images_shape, axis=1)
        end_mask = np.any(end_pix <= 0, axis=1)
        zero_mask = start_mask | end_mask

        # filter out such nodules
        self.nodules = self.nodules[~zero_mask]

    def _rescale_spacing(self):
        """ Rescale spacing values and call _refresh_nodules_info().

        Method is called after any operation that changes shape of inner data.
        """
        if self.nodules is not None:
            self._refresh_nodules_info()
        return self

    def _post_mask(self, list_of_arrs, **kwargs):
        """ Concatenate outputs of different workers and put the result in `masks`

        Parameters
        ----------
        list_of_arrs : list
                       list of ndarrays of patients' masks.
        """
        if any_action_failed(list_of_arrs):
            raise ValueError("Failed while parallelizing")

        new_masks = np.concatenate(list_of_arrs, axis=0)
        self.masks = new_masks

        return self

    def _init_load_blosc(self, **kwargs):
        """ Init-func for load from blosc.

        Fills images/masks-components with zeroes if the components are to be updated.

        Parameters
        ----------
        **kwargs
                src :     str, list or tuple
                          iterable of components names that need to be loaded
        Returns
        -------
        list
            list of ids of batch-items, i.e. series ids or patient ids.
        """
        # read shapes, fill the components with zeroes if masks, images need to
        # be updated
        if 'masks' in kwargs['src'] or 'images' in kwargs['src']:
            slice_shape = self._preload_shapes()
            skysc_shape = (self._bounds[-1], ) + slice_shape

            # fill needed comps with zeroes
            for source in {'images', 'masks'} & set(kwargs['src']):
                setattr(self, source, np.zeros(skysc_shape))

        return self.indices

    def _post_rebuild(self, all_outputs, new_batch=False, **kwargs):
        """ Gather outputs of different workers, rebuild `images` and `masks`.

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
        """ Make intensity projection (maximum, minimum, average) and corresponding masks.

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
        batch
             Resulting batch, where `images` are xips and corresponding `masks`.


        Note
        ----
        Method changes nodules sizes' and creates new `masks` that corresponds
        to data after xip.
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
        """ Make crop of crop_size from center of images.

        Parameters
        ----------
        crop_size : tuple(int, int, int)
            shape of central crop along three axes(z,y,x order is used).
        """
        crop_size = np.asarray(crop_size)
        img_shapes = [np.asarray(self.get(i, 'images').shape)
                      for i in range(len(self))]
        if any(np.any(shape < crop_size) for shape in img_shapes):
            raise ValueError(
                "Crop size must be smaller than size of inner 3D images")

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
        """ Invert the order of slices for each patient

        Example
        -------
        >>> batch = batch.flip()
        """
        logger.warning("There is no implementation of flip method for class " +
                       "CTIMagesMaskedBatch. Nothing happened")
        return self

    @action
    def binarize_mask(self, threshold=0.35):
        """ Binarize masks by threshold.

        Parameters
        ----------
        threshold : float
            threshold for masks binarization.

        """
        self.masks *= np.asarray(self.masks > threshold, dtype=np.int)
        return self

    @action
    def predict_on_scan(self, model_name, strides=(16, 32, 32), crop_shape=(32, 64, 64),
                        batch_size=4, y_component='labels', dim_ordering='channels_last',
                        show_progress=True):
        """ Get predictions of the model on data contained in batch.

        Transforms scan data into patches of shape CROP_SHAPE and then feed
        this patches sequentially into model with name specified by
        argument 'model_name'; after that loads predicted masks or probabilities
        into 'masks' component of the current batch and returns it.

        Parameters
        ----------
        model_name : str
            name of model that will be used for predictions.
        strides : tuple(int, int, int)
            strides for patching operation
        batch_size : int
            number of patches to feed in model in one iteration.
        y_component: str
            name of y component, can be 'masks' or labels.
        dim_ordering: str
            dimension ordering, can be 'channels_first' or 'channels_last'.

        Returns
        -------
        CTImagesMaskedBatch.
        """
        _model = self.get_model_by_name(model_name)

        patches_arr = self.get_patches(patch_shape=crop_shape,
                                       stride=strides,
                                       padding='reflect')
        if dim_ordering == 'channels_first':
            patches_arr = patches_arr[:, np.newaxis, ...]
        elif dim_ordering == 'channels_last':
            patches_arr = patches_arr[..., np.newaxis]

        predictions = []
        iterations = range(0, patches_arr.shape[0], batch_size)
        if show_progress:
            iterations = tqdm_notebook(iterations)  # pylint: disable=redefined-variable-type
        for i in iterations:
            current_prediction = np.asarray(_model.predict(patches_arr[i: i + batch_size, ...]))

            if y_component == 'labels':
                current_prediction = np.stack([np.ones(shape=(crop_shape)) * prob
                                               for prob in current_prediction.ravel()])

            if y_component == 'regression':
                masks_patch = create_mask_reg(current_prediction[:, :3],
                                              current_prediction[:, 3:6],
                                              current_prediction[:, 6],
                                              crop_shape, 0.01)

                current_prediction = np.squeeze(masks_patch)
            predictions.append(current_prediction)

        patches_mask = np.concatenate(predictions, axis=0)
        self.load_from_patches(patches_mask, stride=strides,
                               scan_shape=tuple(self.images_shape[0, :]),
                               data_attr='masks')
        return self

    def unpack_component(self, component='images', dim_ordering='channels_last'):
        """ Basic way for unpacking 'images' or 'masks' from batch.

        Parameters
        ----------
        component : str
            component to unpack, can be 'images' or 'masks'.
        dim_ordering : str
            can be 'channels_last' or 'channels_first'. Reflects where to put
            channels dimension: right after batch dimension or after all spatial axes.

        Returns
        -------
        ndarray(batch_size, zdim, ydim, xdim, 1)
            unpacked 'images' or 'masks' component of batch as numpy array.

        Raises
        ------
        AttributeError
            if argument component is not 'images' or 'masks'.
        """
        if component not in ('masks', 'images'):
            logger.warning("Component must be 'images' or 'masks'. "
                           + "Got {}. Returning None.".format(component))
            return None

        if np.all(self.images_shape == self.images_shape[0, :]):
            x = self.get(None, component).reshape(-1, *self.images_shape[0, :])
        else:
            x = np.stack([self.get(i, component) for i in range(len(self))])

        if dim_ordering == 'channels_last':
            x = x[..., np.newaxis]
        elif dim_ordering == 'channels_first':
            x = x[:, np.newaxis, ...]
        return x

    def _unpack_seg(self, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for segmentation task.

        Parameters
        ----------
        dim_ordering : str
            can be 'channels_last' or 'channels_first'. Reflects where to put
            channels dimension: right after batch dimension or after all spatial axes.

        Returns
        -------
        dict
            {'inputs': images_array, 'targets': labels_array}

        Note
        ----
        'dim_ordering' argument reflects where to put '1'
        for channels dimension both for images and masks.
        """
        return {'inputs': self.unpack_component('images', dim_ordering),
                'targets': self.unpack_component('masks', dim_ordering)}

    def _unpack_clf(self, threshold=10, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for classification task.

        Parameters
        ----------
        threshold : int
            minimum number of '1' pixels in mask to consider it cancerous.
        dim_ordering : str
            can be 'channels_last' or 'channels_first'. Reflects where to put
            channels dimension: right after batch dimension or after all spatial axes.

        Returns
        -------
        dict
            {'inputs': images_array, 'targets': labels_array}

        Note
        ----
        'dim_ordering' argument reflects where to put '1' for channels dimension.
        """
        masks_labels = np.asarray([self.get(i, 'masks').sum() > threshold
                                   for i in range(len(self))], dtype=np.int)

        return {'inputs': self.unpack_component('images', dim_ordering),
                'targets': masks_labels[:, np.newaxis]}

    def _unpack_reg(self, threshold=10, dim_ordering='channels_last'):
        """ Unpack data from batch in format suitable for regression task.

        Parameters
        ----------
        threshold : int
            minimum number of '1' pixels in mask to consider it cancerous.

        Returns
        -------
        dict
            {'inputs': images_array, 'targets': y_regression_array}

        Note
        ----
        'dim_ordering' argument reflects where to put '1' for channels dimension.

        """

        nodules = self.nodules

        sizes = np.zeros(shape=(len(self), 3), dtype=np.float)
        centers = np.zeros(shape=(len(self), 3), dtype=np.float)

        for item_pos, _ in enumerate(self.indices):
            item_nodules = nodules[nodules.item_pos == item_pos]

            if len(item_nodules) == 0:
                continue

            mask_nod_indices = item_nodules.nodule_size.max(axis=1).argmax()

            nodule_sizes = (item_nodules.nodule_size / self.spacing[item_pos, :]
                            / self.images_shape[item_pos, :])

            nodule_centers = (item_nodules.nodule_center / self.spacing[item_pos, :]
                              / self.images_shape[item_pos, :])

            sizes[item_pos, :] = nodule_sizes[mask_nod_indices, :]
            centers[item_pos, :] = nodule_centers[mask_nod_indices, :]

        clf_dict = self.unpack_clf(threshold, dim_ordering)
        x, labels = clf_dict['inputs'], clf_dict['targets']
        y_regression_array = np.concatenate([centers, sizes, labels], axis=1)

        return {'inputs': x, 'targets': y_regression_array}

    @action
    def set_targets(self, mode='segmentation', **kwargs):
        """ Prepare data in batch for training neural network.

        Parameters
        ----------
        mode : str
            mode can be one of following 'classification', 'regression'
            or 'segmentation'.
        """
        if mode == 'segmentation':
            feed_dict = self._unpack_seg(**kwargs)
        elif mode == 'regression':
            feed_dict = self._unpack_reg(**kwargs)
        elif mode == 'classification':
            feed_dict = self._unpack_clf(**kwargs)
        else:
            return self

        self.targets = feed_dict['targets']
        self.inputs = feed_dict['inputs']
        return self
