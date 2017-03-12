# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path)
from dataset import * # pylint: disable=wrong-import-

# Create index from ./data
findex = FilesIndex('./data')
# print list of files
print("Index:")
print(findex.index)

print("\nSplit")
findex.cv_split([0.35, 0.35])
for dsi in [findex.train, findex.test, findex.validation]:
    if dsi is not None:
        print(dsi.index)

print("\nprint batches:")
for dsi in [findex.train, findex.test, findex.validation]:
    print("---")
    for b in dsi.gen_batch(2, one_pass=True):
        print(b)


# Create index from ./data/dirs
dindex = DirectoriesIndex('./data/dirs', sort=True)
# print list of subdirectories
print("Index:")
print(dindex.index)

print("\nSplit")
dindex.cv_split([0.35, 0.35])
for dsi in [dindex.train, dindex.test, dindex.validation]:
    if dsi is not None:
        print(dsi.index)
