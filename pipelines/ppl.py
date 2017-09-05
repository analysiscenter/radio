""" Helper functions describing pipelines for creating large samples of nodules """

import PIL

# global constants defining args of some actions in pipeline

SPACING = (1.7, 1.0, 1.0)
USPACING_SHAPE = (400, 512, 512)
RESIZE_FILTER = PIL.Image.LANCZOS

# define the number of times each cancerous nodule is dumped.
# with this number of iterations, the whole luna-dataset will
# produce approximately 115000 cancerous crops
N_ITERS = 100

# these params ensure that the number of non-cancerous crops will also
# be around 115000 (when run on the whole luna-dataset)
RUN_BATCH_SIZE = 8
NCANCER_BATCH_SIZE = 1030


def split_dump_lunaset(lunaset, dir_cancer, dir_ncancer, nodules_df, histo, nodule_shape=(32, 64, 64),
                       variance=(36, 144, 144)):
    """ Define pipeline for dumping cancerous crops in one folder
            and random noncancerous crops in another.

    Args:
        lunaset: dataset of luna scans.
        dir_cancer: directory to dump cancerous crops in.
        dir_ncancer: directory to dump non-cancerous crops in.
        nodules_df: df with info about nodules' locations.
        histo: distribution in np.histogram format that is used for sampling
            non-cancerous crops.
        nodule_shape: shape of crops.
        variance: variance of locations of cancerous nodules' centers in generated
            cancerous crops.

    Return:
        resulting pipeline run in lazy-mode
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
    # 2) sampling and dumping of non-cancerous crops in separete actions
    pipeline = (lunaset.p
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
