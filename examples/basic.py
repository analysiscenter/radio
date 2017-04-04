#pylint: skip-file
import os
import sys
import glob
sys.path.append('..')
from preprocessing import CTImagesBatch
from dataset import FilesIndex, Dataset, action


class ExampleCTIBatch(CTImagesBatch):
    @action
    def load(self, src, fmt='dicom', *args, **kwargs): # pylint: disable=arguments-differ
        if fmt == 'some':
            list_of_arrs = self._load_some()
        else:
        	list_of_arrs = super().load(src, fmt, *args, **kwargs)
        return self

    def _load_some(self):
        list_of_arrs = []
        for patient in self.indices:
            patient_folder = self.index.get_fullpath(patient)
            list_of_files = [s for s in os.listdir(patient_folder)]
            print("--- files in", patient_folder)
            print("   ", list_of_files)
        return list_of_arrs


DIR_TEST = '../../_tests/_data/nci/*'

ct_index = FilesIndex(path=DIR_TEST, dirs=True)
print("Full index:", ct_index.index)

ct_ds = Dataset(ct_index, batch_class=ExampleCTIBatch)

print()
for ct_batch in ct_ds.gen_batch(2, shuffle=False, one_pass=True):
	ctb = ct_batch.load(DIR_TEST, fmt='some')
	print("Index:", ctb.index)
	print("      ", ctb.indices)
