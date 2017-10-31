""" Helper functions describing pipelines for creating large samples of nodules """

import PIL
from ..dataset import Pipeline

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
RUN_BATCH_SIZE = 8
NON_CANCER_BATCH_SIZE = 1030  # NON_CANCER_BATCH_SIZE * (len_of_lunaset=888) / RUN_BATCH_SIZE ~ 115000


def get_crops(nodules_df, fmt='raw', nodule_shape=(32, 64, 64), batch_size=20, share=0.5, histo=None,
              variance=(36, 144, 144), hu_lims=(-1000, 400), **kwargs):
    """ Define pipeline that performs preprocessing and crops cancerous/non-cancerous nodules in
            a chosen proportion.

    Args:
        nodules_df: df with info about nodules' locations.
        fmt: format of scans ('raw'|'blosc'|'dicom').
        nodule_shape: shape of crops.
        batch_size: size of generated batch.
        share: share of cancerous crops in a generated batch.
        histo: distribution in np.histogram format that is used for sampling
            non-cancerous crops.
        variance: variance of locations of cancerous nodules' centers in generated
            cancerous crops.
        hu_lims: seq of len=2, representing limits of hu-trimming in normalize_hu-action.
        kwargs: unify_spacing-params that need to be changed from default values (see kwargs_default).

    Return:
        resulting pipeline run in lazy-mode.
    """
    # update args of unify spacing
    kwargs_default.update(kwargs)

    # set up other args-dicts
    args_sample_nodules = dict(nodule_size=nodule_shape, batch_size=batch_size, share=share,
                               histo=histo, variance=variance)

    # set up the pipeline
    pipeline = (Pipeline()
                .load(fmt=fmt)
                .fetch_nodules_info(nodules_df=nodules_df)
                .unify_spacing(**kwargs_default)
                .create_mask()
                .normalize_hu(min_hu=hu_lims[0], max_hu=hu_lims[1])
                .sample_nodules(**args_sample_nodules)
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=True)
               )

    return pipeline


def split_dump(cancer_path, non_cancer_path, nodules_df, histo=None, fmt='raw', nodule_shape=(32, 64, 64),
               variance=(36, 144, 144), **kwargs):
    """ Define pipeline for dumping cancerous crops in one folder
            and random noncancerous crops in another.

    Args:
        cancer_path: directory to dump cancerous crops in.
        non_cancer_path: directory to dump non-cancerous crops in.
        nodules_df: df with info about nodules' locations.
        histo: distribution in np.histogram format that is used for sampling
            non-cancerous crops.
        fmt: format of scans ('raw'|'blosc'|'dicom').
        nodule_shape: shape of crops.
        variance: variance of locations of cancerous nodules' centers in generated
            cancerous crops.
        kwargs: unify_spacing-params that need to be changed from default values (see kwargs_default).

    Return:
        resulting pipeline run in lazy-mode.
    """
    # update args of unify spacing
    kwargs_default.update(kwargs)

    # set up args-dicts
    args_dump_cancer = dict(dst=cancer_path, n_iters=N_ITERS, nodule_size=nodule_shape, variance=variance,
                            share=1.0, batch_size=None)
    args_sample_ncancer = dict(nodule_size=nodule_shape, histo=histo, batch_size=NON_CANCER_BATCH_SIZE, share=0.0)

    # define pipeline. Two separate tasks are performed at once, in one run:
    # 1) sampling and dumping of cancerous crops in wrapper-action sample_sump_cancerous
    # 2) sampling and dumping of non-cancerous crops in separate actions
    pipeline = (Pipeline()
                .load(fmt=fmt)
                .fetch_nodules_info(nodules_df=nodules_df)
                .unify_spacing(**kwargs_default)
                .create_mask()
                .sample_dump(**args_dump_cancer)  # sample and dump cancerous crops
                .sample_nodules(**args_sample_ncancer)  # sample non-cancerous
                .dump(dst=non_cancer_path)  # dump non-cancerous
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=False)
               )

    return pipeline

def update_histo(nodules_df, histo, fmt='raw', **kwargs):
    """ Pipeline for updating histogram using info in dataset of scans.

    Args:
        nodules_df: df with info about nodules' locations.
        histo: 3d-histogram in almost np.histogram format, list [bins, edges];
            (compare the latter with tuple (bins, edges) returned by np.histogram).
        fmt: format of scans ('raw'|'blosc'|'dicom').
        kwargs: unify_spacing-params that need to be changed from default values (see kwargs_default).

    Return:
        resulting pipeline run in lazy-mode.
    """
    # update args of unify spacing
    kwargs_default.update(kwargs)

    # perform unify_spacing and call histo-updating action
    pipeline = (Pipeline()
                .load(fmt=fmt)
                .fetch_nodules_info(nodules_df=nodules_df)
                .unify_spacing(**kwargs_default)
                .create_mask()
                .update_nodules_histo(histo)
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=False)
               )

    return pipeline

def combine_crops(cancerset, ncancerset, batch_sizes=(10, 10), hu_lims=(-1000, 400)):
    """ Pipeline for generating batches of cancerous and non-cancerous crops from
            ct-scans in chosen proportion.

    Args:
        cancerset: dataset of cancerous crops in blosc format.
        ncancerset: dataset of non-cancerous crops in blosc format.
        batch_sizes: seq of len=2, (num_cancer_batches, num_noncancer_batches).
        hu_lims: seq of len=2, representing limits of hu-trimming in normalize_hu-action.

    Return:
        lazy-run pipeline, that will generate batches of size = batch_sizes[0] + batch_sizes[1],
            when run without arguments.
    """
    # pipeline generating cancerous crops
    ppl_cancer = (cancerset.p
                  .load(fmt='blosc')
                  .normalize_hu(min_hu=hu_lims[0], max_hu=hu_lims[1])
                  .run(lazy=True, batch_size=batch_sizes[0], shuffle=True)
                 )

    # pipeline generating non-cancerous crops merged with first pipeline
    pipeline = (ncancerset.p
                .load(fmt='blosc')
                .normalize_hu(min_hu=hu_lims[0], max_hu=hu_lims[1])
                .merge(ppl_cancer)
                .run(lazy=True, batch_size=batch_sizes[1], shuffle=True)
               )

    return pipeline
