""" Helper functions describing pipelines for creating large samples of nodules """

import PIL

def dump_cancerous(lunaset, dir_dump, nodules_df):
	""" Define pipeline for dumping only (and all) cancerous nodules

	Args:
		lunaset: dataset of luna scans
		dir_dump: directory to dump cancerous crops in
		nodules_df: df with info about nodules' locations

	Return:
		resulting pipeline
	"""
	# set up all args
	args_load = dict(fmt='raw')
	args_fetch = dict(nodules_df=nodules_df)
	args_mask = dict()
	args_unify_spacing = dict(spacing=(1.7, 1.0, 1.0), shape=(400, 512, 512), padding='reflect',
							  resample=PIL.Image.LANCZOS)
	args_sample_dump = dict(dst=dir_dump, n_iters=100, nodule_size=(32, 64, 64), variance=(36, 144, 144))

	# define pipeline
	pipeline = (lunaset.p
				.load(**args_load)
				.fetch_nodules_info(**args_fetch)
				.unify_spacing(**args_unify_spacing)
				.create_mask(**args_mask)
				.sample_dump_cancerous(**args_sample_dump)
			   )

	return pipeline
