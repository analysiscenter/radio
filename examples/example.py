import sys

sys.path.append('..')
from radio import CTImagesMaskedBatch as CTIMB
from radio.dataset import Pipeline, B, V, F, FilesIndex, Dataset

def load_example(path=None, fmt='blosc'):
    if path is None:
        path = '../../scans_sample/1.3.6.1.4.1.14519.5.2.1.6279.6001.621916089407825046337959219998'
    luna_index = FilesIndex(path=path, dirs=True)
    lunaset =  Dataset(luna_index, batch_class=CTIMB)
    load_ppl = (Pipeline()
                 .load(fmt='blosc', components=['images', 'spacing', 'origin', 'masks']) << lunaset)

    btch = load_ppl.next_batch(1)
    return btch
