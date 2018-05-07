""" Functions for processing annotation provided for dicom dataset. """

from .parser import read_nodules, read_dataset_info, read_annotators_info
from .nodules_merger import assign_nodules_group_index, get_nodules_groups
from .nodules_merger import compute_group_coords_and_diameter
from .doctor_confidence import get_doctors_confidences
from .doctor_confidence import get_probabilities, get_rating, get_common_annotation, get_table
from .nodule_confidence import compute_nodule_confidence
