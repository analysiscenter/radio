""" Helper functions describing pipelines for creating large samples of nodules """

import PIL
from ..dataset import Pipeline

# global constants defining args of some actions in pipeline

SPACING = (1.7, 1.0, 1.0)
USPACING_SHAPE = (400, 512, 512)
RESIZE_FILTER = PIL.Image.LANCZOS

# define the number of times each cancerous nodule is dumped.
# with this number of iterations, the whole luna-dataset will
# produce approximately 115000 cancerous crops
N_ITERS = 100  # N_ITERS * (num_luna_nodules=1149) ~ 115000

# these params ensure that the number of non-cancerous crops will also
# be around 115000 (when run on the whole luna-dataset)
RUN_BATCH_SIZE = 8
NCANCER_BATCH_SIZE = 1030 # NCANCER_BATCH_SIZE * (len_of_lunaset=888) / RUN_BATCH_SIZE ~ 115000


def split_dump_lunaset(dir_cancer, dir_ncancer, nodules_df, histo, nodule_shape=(32, 64, 64),
                       variance=(36, 144, 144)):
    """ Define pipeline for dumping cancerous crops in one folder
            and random noncancerous crops in another.

    Args:
        dir_cancer: directory to dump cancerous crops in.
        dir_ncancer: directory to dump non-cancerous crops in.
        nodules_df: df with info about nodules' locations.
        histo: distribution in np.histogram format that is used for sampling
            non-cancerous crops.
        nodule_shape: shape of crops.
        variance: variance of locations of cancerous nodules' centers in generated
            cancerous crops.

    Return:
        resulting pipeline run in lazy-mode.
    """
    # set up all args
    args_load = dict(fmt='raw')
    args_fetch = dict(nodules_df=nodules_df)
    args_mask = dict()
    args_unify_spacing = dict(spacing=SPACING, shape=USPACING_SHAPE, padding='reflect', resample=RESIZE_FILTER)
    args_dump_cancer = dict(dst=dir_cancer, n_iters=N_ITERS, nodule_size=nodule_shape, variance=variance)
    args_sample_ncancer = dict(nodule_size=nodule_shape, histo=histo, batch_size=NCANCER_BATCH_SIZE, share=0.0)
    args_dump_ncancer = dict(dst=dir_ncancer)

    # define pipeline. Two separate tasks are performed at once, in one run:
    # 1) sampling and dumping of cancerous crops in wrapper-action sample_sump_cancerous
    # 2) sampling and dumping of non-cancerous crops in separate actions
    pipeline = (Pipeline()
                .load(**args_load)
                .fetch_nodules_info(**args_fetch)
                .unify_spacing(**args_unify_spacing)
                .create_mask(**args_mask)
                .sample_dump_cancerous(**args_dump_cancer)   # sample and dump non-cancerous crops
                .sample_nodules(**args_sample_ncancer)
                .dump(**args_dump_ncancer)
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=False)
               )

    return pipeline

def update_histo_by_lunaset(nodules_df, histo):
    """ Pipeline for updating histogram using info in luna-dataset.

    Args:
        nodules_df: df with info about nodules' locations.
        histo: 3d-histogram in almost np.histogram format, list [bins, edges];
            (compare the latter with tuple (bins, edges) returned by np.histogram).

    Return:
        resulting pipeline run in lazy-mode.
    """
    # set up all args
    args_load = dict(fmt='raw')
    args_fetch = dict(nodules_df=nodules_df)
    args_unify_spacing = dict(spacing=SPACING, shape=USPACING_SHAPE, padding='reflect', resample=RESIZE_FILTER)
    args_mask = dict()

    # perform unify_spacing and call histo-updating action
    pipeline = (lunaset.p
                .load(**args_load)
                .fetch_nodules_info(**args_fetch)
                .unify_spacing(**args_unify_spacing)
                .create_mask(**args_mask)
                .update_nodules_histo(histo)
                .run(lazy=True, batch_size=RUN_BATCH_SIZE, shuffle=False)
               )

    return pipeline

def get_luna_crops(cancerset, ncancerset, batch_sizes=(10, 10), hu_lims=(-1000, 400)):
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
    # set up args of all actions
    args_load = dict(fmt='blosc')
    args_normalize_hu = dict(min_hu=hu_lims[0], max_hu=hu_lims[1])

    # pipeline generating cancerous crops
    ppl_cancer = (cancerset.p
                  .load(**args_load)
                  .normalize_hu(**args_normalize_hu)
                  .run(lazy=True, batch_size=batch_sizes[0], shuffle=True)
                 )

    # pipeline generating non-cancerous crops merged with first pipeline
    pipeline = (ncancerset.p
                .load(**args_load)
                .normalize_hu(**args_normalize_hu)
                .merge(ppl_cancer)
                .run(lazy=True, batch_size=batch_sizes[1], shuffle=True)
               )

    return pipeline
