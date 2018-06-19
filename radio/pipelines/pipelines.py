""" Helper functions describing pipelines for creating large samples of nodules """

from copy import copy
import PIL
from .pipelines import options_seq, options_prod
from ..dataset import Pipeline  # pylint: disable=no-name-in-module

# global constants defining args of some actions in pipeline
SPACING = (1.7, 1.0, 1.0)  # spacing of scans after spacing unification
SHAPE = (400, 512, 512)  # shape of scans after spacing unification
RESIZE_FILTER = PIL.Image.LANCZOS  # high-quality filter of resize
PADDING = 'reflect'  # padding-mode that produces the least amount of artefacts
METHOD = 'pil-simd'  # robust resize-engine
kwargs_default = dict(shape=SHAPE, spacing=SPACING, resample=RESIZE_FILTER, padding=PADDING, method=METHOD)

# define the number of times each cancerous nodule is dumped.
# with this number of iterations, the whole luna-dataset will
# produce approximately 115000 cancerous crops
N_ITERS = 100  # N_ITERS * (num_luna_nodules=1149) ~ 115000

# these params ensure that the number of non-cancerous crops will also
# be around 115000 (when run on the whole luna-dataset)
RUN_BATCH_SIZE = 6
NON_CANCER_BATCH_SIZE = 800  # NON_CANCER_BATCH_SIZE * (len_of_lunaset=888) / RUN_BATCH_SIZE ~ 115000


def get_crops(nodules, fmt='raw', nodule_shape=(32, 64, 64), batch_size=20, share=0.5, histo=None,
              variance=(36, 144, 144), hu_lims=(-1000, 400), **kwargs):
    """ Get pipeline that performs preprocessing and crops cancerous/non-cancerous nodules in
    a chosen proportion.

    Parameters
    ----------
    nodules : pd.DataFrame
        contains:
         - 'seriesuid': index of patient or series.
         - 'z','y','x': coordinates of nodules center.
         - 'diameter': diameter, in mm.
    fmt : str
        can be either 'raw', 'blosc' or 'dicom'.
    nodule_shape : tuple, list or ndarray of int
        crop shape along (z,y,x).
    batch_size : int
        number of nodules in batch generated by pipeline.
    share : float
        share of cancer crops in the batch.
    histo : tuple
        :func:`numpy.histogramdd` output.
        Used for sampling non-cancerous crops
    variance : tuple, list or ndarray of float
        variances of normally distributed random shifts of
        nodules' start positions
    hu_lims : tuple, list of float
        seq of len=2, representing limits of hu-trimming in normalize_hu-action.
    **kwargs
            spacing : tuple
                (z,y,x) spacing after resize.
            shape : tuple
                (z,y,x) shape after crop/pad.
            method : str
                interpolation method ('pil-simd' or 'resize').
                See :func:`~radio.CTImagesBatch.resize`.
            order : None or int
                order of scipy-interpolation (<=5), if used.
            padding : str
                mode of padding, any supported by :func:`numpy.pad`.

    Returns
    -------
    pipeline
    """
    # update args of unify spacing
    args_unify_spacing = copy(kwargs_default)
    args_unify_spacing.update(kwargs)

    # set up other args-dicts
    args_sample_nodules = dict(nodule_size=nodule_shape, batch_size=batch_size, share=share,
                               histo=histo, variance=variance)

    # set up the pipeline
    pipeline = (Pipeline()
                .load(fmt=fmt)
                .fetch_nodules_info(nodules=nodules)
                .unify_spacing(**args_unify_spacing)
                .create_mask()
                .normalize_hu(min_hu=hu_lims[0], max_hu=hu_lims[1])
                .sample_nodules(**args_sample_nodules)
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=True)
               )

    return pipeline


def split_dump(cancer_path, non_cancer_path, nodules, histo=None, fmt='raw',
               nodule_shape=(32, 64, 64), variance=(36, 144, 144), **kwargs):
    """ Get pipeline for dumping cancerous crops in one folder and random noncancerous crops in another.

    Parameters
    ----------
    cancer_path : str
        directory to dump cancerous crops in.
    non_cancer_path : str
        directory to dump non-cancerous crops in.
    nodules : pd.DataFrame
        contains:
         - 'seriesuid': index of patient or series.
         - 'z','y','x': coordinates of nodules center.
         - 'diameter': diameter, in mm.
    histo : tuple
        :func:`numpy.histogramdd` output.
        Used for sampling non-cancerous crops
    fmt : str
        can be either 'raw', 'blosc' or 'dicom'.
    nodule_shape : tuple, list or ndarray of int
        crop shape along (z,y,x).
    variance : tuple, list or ndarray of float
        variances of normally distributed random shifts of
        nodules' start positions
    **kwargs
            spacing : tuple
                (z,y,x) spacing after resize.
            shape : tuple
                (z,y,x) shape after crop/pad.
            method : str
                interpolation method ('pil-simd' or 'resize').
                See :func:`~radio.CTImagesBatch.resize` for more information.
            order : None or int
                order of scipy-interpolation (<=5), if used.
            padding : str
                mode of padding, any supported by :func:`numpy.pad`.

    Returns
    -------
    pipeline
    """
    # update args of unify spacing
    args_unify_spacing = copy(kwargs_default)
    args_unify_spacing.update(kwargs)

    # set up args-dicts
    args_dump_cancer = dict(dst=cancer_path, n_iters=N_ITERS, nodule_size=nodule_shape,
                            variance=variance, share=1.0, batch_size=None)
    args_sample_ncancer = dict(nodule_size=nodule_shape, histo=histo,
                               batch_size=NON_CANCER_BATCH_SIZE, share=0.0)

    # define pipeline. Two separate tasks are performed at once, in one run:
    # 1) sampling and dumping of cancerous crops in wrapper-action sample_sump_cancerous
    # 2) sampling and dumping of non-cancerous crops in separate actions
    pipeline = (Pipeline()
                .load(fmt=fmt)
                .fetch_nodules_info(nodules=nodules)
                .unify_spacing(**args_unify_spacing)
                .create_mask()
                .sample_dump(**args_dump_cancer)  # sample and dump cancerous crops
                .sample_nodules(**args_sample_ncancer)  # sample non-cancerous
                .dump(dst=non_cancer_path)  # dump non-cancerous
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=False)
               )

    return pipeline

def update_histo(nodules, histo, fmt='raw', **kwargs):
    """ Pipeline for updating histogram using info in dataset of scans.

    Parameters
    ----------
    nodules : pd.DataFrame
        contains:
         - 'seriesuid': index of patient or series.
         - 'z','y','x': coordinates of nodules center.
         - 'diameter': diameter, in mm.
    histo : tuple
        :func:`numpy.histogramdd` output.
        Used for sampling non-cancerous crops
        (compare the latter with tuple (bins, edges) returned by :func:`numpy.histogramdd`).
    fmt : str
        can be either 'raw', 'blosc' or 'dicom'.
    **kwargs
            spacing : tuple
                (z,y,x) spacing after resize.
            shape : tuple
                (z,y,x) shape after crop/pad.
            method : str
                interpolation method ('pil-simd' or 'resize').
                See :func:`~radio.CTImagesBatch.resize` for more information.
            order : None or int
                order of scipy-interpolation (<=5), if used.
            padding : str
                mode of padding, any supported by :func:`numpy.pad`.

    Returns
    -------
    pipeline
    """
    # update args of unify spacing
    args_unify_spacing = copy(kwargs_default)
    args_unify_spacing.update(kwargs)

    # perform unify_spacing and call histo-updating action
    pipeline = (Pipeline()
                .load(fmt=fmt)
                .fetch_nodules_info(nodules=nodules)
                .unify_spacing(**args_unify_spacing)
                .create_mask()
                .update_nodules_histo(histo)
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=False)
               )

    return pipeline

def combine_crops(cancer_set, non_cancer_set, batch_sizes=(10, 10), hu_lims=(-1000, 400),
                  components=('images', 'masks', 'origin', 'spacing')):
    """ Pipeline for generating batches of cancerous and non-cancerous crops from
    ct-scans in chosen proportion.

    Parameters
    ---------
    cancer_set : dataset
        dataset of cancerous crops in blosc format.
    non_cancer_set : dataset
        dataset of non-cancerous crops in blosc format.
    batch_sizes : tuple, list of int
        seq of len=2, (num_cancer_batches, num_noncancer_batches).
    hu_lims : tuple, list of float
        seq of len=2, representing limits of hu-trimming in normalize_hu-action.
    components : tuple, list of str
        components to load.

    Returns
    -------
    pipeline
    """
    # pipeline generating cancerous crops
    ppl_cancer = (cancer_set.p
                  .load(fmt='blosc', components=components)
                  .normalize_hu(min_hu=hu_lims[0], max_hu=hu_lims[1])
                  .run(lazy=True, batch_size=batch_sizes[0], shuffle=9)
                 )

    # pipeline generating non-cancerous crops merged with first pipeline
    pipeline = (non_cancer_set.p
                .load(fmt='blosc', components=components)
                .normalize_hu(min_hu=hu_lims[0], max_hu=hu_lims[1])
                .merge(ppl_cancer)
                .run(lazy=True, batch_size=batch_sizes[1], shuffle=9)
               )

    return pipeline


def cancer_rate_to_num_repeats(batch, rate):
    """ Compute number of times to run batch through augmentation pipeline.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        batch with fetched info about nodules location.
    rate : float
        number of samples with nodules per scan.

    Returns
    -------
    int
        number of times to run batch through augmentation pipeline
        required to get given average per scan sample rate.
    """
    if batch.nodules is None:
        return 1
    else:
        return math.ceil(rate * len(batch) / len(batch.nodules))


def ncancer_rate_to_num_repeats(batch, rate, max_crops):
    """ Compute number of times to run batch through augmentation pipeline.

    Parameters
    ----------
    batch : CTImagesMaskedBatch
        batch with fetched info about nodules location.
    rate : float
        number of samples with nodules per scan.

    Returns
    -------
    int
        number of times to pass batch through augmentation pipeline
        required to get given average per scan sample rate.
    """
    return math.ceil(rate * len(batch) / max_crops)


@options_prod(angle=(-15, 15, -30, 30, -45, 45, -60, 60, -90, 90))
@options_seq(cancerous=(True, False), suffix=('cancer/rotation', 'ncancer/rotation'))
def sample_and_rotate(crop_size=(32, 64, 64), histo=None, variance=(49, 169, 169),
                      angle=30, rot_axes=(1, 2), rate=16, max_crops=64,
                      cancerous=True, dst=None, suffix=None):
    """ Sample cancerous and non-cancerous and rotate them to 10 different angles.

    Samples are rotated on angles (-15, 15, -30, 30, -45, 45, -60, 60, -90, 90).

    Parameters
    ----------
    crop_size : ArrayLike(int)
        size of crop along (z, y, x) axes.
    histo : tuple(ndarray, ndarray)
        histogramm that will be used for sampling non-cancerous crops.
    variance : ArrayLike(int)
        variance along (z, y, x) axes.
    rot_axes : ArrayLike(int)
        rotation plane defined by two axes.
    rate : int
        per scan expected number of samples. This parameter is
        considered to be shared for both cancerous and non-cancerous cases.
    max_crops : int
        maximum number of crops for non-cancerous sampling.
    cancerous : bool
        whether to sample cancerous or non-cancerous crops.
    dst : str or None
        if None then crops are not dumped otherwise must be a string
        representing path to dump directory.
    """

    nodule_size = np.rint(np.array(crop_size) * np.sqrt(2)).astype(np.int)
    batch_size = max_crops if not cancerous else None
    pipeline = (
        ds.Pipeline()
        .sample_nodules(share=int(cancerous), nodule_size=nodule_size,
                        variance=variance, batch_size=batch_size, histo=histo)
        .rotate(angle=angle, axes=rot_axes, random=False, inplace=False)
        .create_mask()
        .central_crop(size=crop_size, crop_mask=True, inplace=False)
    )

    if dst is not None:
        _path = os.path.join(dst, suffix) if suffix else str(dst)
        pipeline = pipeline.dump(dst=_path)

    if cancerous is None:
        return (
            ds.Pipeline()
            .repeat_pipeline(pipeline, F(cancer_rate_to_num_repeats, rate=rate))
        )
    else:
        return (
            ds.Pipeline()
            .repeat_pipeline(pipeline, F(ncancer_rate_to_num_repeats,
                                         rate=rate, max_crops=max_crops))
        )


@repeat_pipeline(num_repeats=10)
@options_seq(cancerous=(True, False), suffix=('cancer/rnd_resize', 'ncancer/rnd_resize'))
def sample_random_resize(crop_size=(32, 64, 64), histo=None, variance=(49, 169, 169),
                         size_percent=0.15, rate=160, max_crops=64, cancerous=True,
                         dump=False, dst=None, suffix=None):
    """ Sample cancerous and non-cancerous and resize them to radom size.

    Parameters
    ----------
    crop_size : ArrayLike(int)
        size of crop along (z, y, x) axes.
    histo : tuple(ndarray, ndarray)
        histogramm that will be used for sampling non-cancerous crops.
    variance : ArrayLike(int)
        variance along (z, y, x) axes.
    size_percent : float
        scatter of size defined as percents of original size.
    rate : int
        per scan expected number of samples. This parameter is
        considered to be shared for both cancerous and non-cancerous cases.
    max_crops : int
        maximum number of crops for non-cancerous sampling.
    cancerous : bool
        whether to sample cancerous or non-cancerous crops.
    dst : str or None
        if None then crops are not dumped otherwise must be a string
        representing path to dump directory.
    """

    nodule_size = np.rint(np.array(crop_size) * np.sqrt(2)).astype(np.int)
    batch_size = max_crops if not cancerous else None

    pipeline = (
        ds.Pipeline()
        .sample_nodules(share=int(cancerous), nodule_size=nodule_size,
                        variance=variance, batch_size=batch_size, histo=histo)
        .resize(shape=F(sample_shape, size_percent=size_percent),
                method='scipy', order=3)
        .central_crop(size=crop_size, crop_mask=True, inplace=False)
    )

    if dst is not None:
        _path = os.path.join(dst, suffix) if suffix else str(dst)
        pipeline = pipeline.dump(dst=_path)

    if cancerous is None:
        return (
            ds.Pipeline()
            .repeat_pipeline(pipeline, F(cancer_rate_to_num_repeats,
                                         rate=rate / 10))
        )
    else:
        return (
            ds.Pipeline()
            .repeat_pipeline(pipeline, F(ncancer_rate_to_num_repeats,
                                         rate=rate / 10, max_crops=max_crops))
        )


@options_seq(cancerous=(True, False), suffix=('cancer/original', 'ncancer/original'))
def sample_simple(crop_size=(32, 64, 64), histo=None, variance=(49, 169, 169),
                  rate=160, max_crops=64, cancerous=True, dst=None, suffix=None):
    """ Sample cancerous and non-cancerous.

    Parameters
    ----------
    crop_size : ArrayLike(int)
        size of crop along (z, y, x) axes.
    histo : tuple(ndarray, ndarray)
        histogramm that will be used for sampling non-cancerous crops.
    variance : ArrayLike(int)
        variance along (z, y, x) axes.
    rate : float
        per scan expected number of samples. This parameter is
        considered to be shared for both cancerous and non-cancerous cases.
    max_crops : int
        maximum number of crops for non-cancerous sampling.
    cancerous : bool
        whether to sample cancerous or non-cancerous crops.
    dst : str or None
        if None then crops are not dumped otherwise must be a string
        representing path to dump directory.
    """

    batch_size = max_crops if not cancerous else None
    pipeline = (
        ds.Pipeline()
        .sample_nodules(share=int(cancerous), nodule_size=crop_size,
                        variance=variance, batch_size=batch_size, histo=histo)
    )

    if dst is not None:
        _path = os.path.join(dst, suffix) if suffix else str(dst)
        pipeline = pipeline.dump(dst=_path)

    if cancerous is None:
        return (
            ds.Pipeline()
            .repeat_pipeline(pipeline, F(cancer_rate_to_num_repeats, rate=rate))
        )
    else:
        return (
            ds.Pipeline()
            .repeat_pipeline(pipeline, F(ncancer_rate_to_num_repeats,
                                         rate=rate, max_crops=max_crops))
        )
