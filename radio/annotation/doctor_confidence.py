#pylint:disable=not-an-iterable
#pylint:disable=cell-var-from-loop

""" Functions to compute doctors' confidences from annotation. """

import os
from numba import jit, prange, njit
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
import multiprocessing as mp
from . import read_nodules, read_dataset_info


N_DOCTORS = 15


def compute_confidences(nodules, confidences='random', n_iters=25, n_consiliums=10, factor=0.3, history=False):
    """ Conpute confidences for doctors

    Parameters
    ----------
    nodules : pd.DataFrame

    confidences : str or list of len N_DOCTORS
        if 'random', initial confidences will be sampled
        if 'uniform', initial confidences is a uniform distribution
        if list, confidences of doctors
    n_iters : int
        number of iterations of updating algorithm
    n_consiliums : int
        number of consiliums for each doctor
    factor : float
        ratio for mask creation
    history : bool
        if False, the function returns final confidences, if True, all history of updating confidences

    Returns
    -------
    new_confidences : np.ndarray
    """
    probabilities = get_probabilities(nodules)
    if confidences == 'random':
        confidences = np.ones(N_DOCTORS) + np.random.uniform(0, 0.1, N_DOCTORS)
        confidences = confidences / np.sum(confidences)
    elif confidences == 'uniform':
        confidences = np.ones(N_DOCTORS) / N_DOCTORS
    confidences_history = [pd.DataFrame({'DoctorID': [str(i).zfill(3) for i in range(N_DOCTORS)],
                                         'confidence': confidences, 'iteration': 0})]
    for i in tqdm(range(n_iters)):
        confidences = update_confidences(nodules, confidences, probabilities, n_consiliums, factor)
        confidences_history.append(pd.DataFrame({'DoctorID': [str(i).zfill(3) for i in range(N_DOCTORS)],
                                                 'confidence': confidences, 'iteration': i+1}))
    return pd.concat(confidences_history, axis=0) if history else confidences_history[-1].drop(columns=['iteration'])


def update_confidences(nodules, confidences, probabilities, n_consiliums=10, factor=0.3, alpha=0.7):
    nodules = (
        nodules
        .assign(n_annotators=lambda df: df.filter(regex=r'doctor_\d{3}', axis=1).sum(axis=1))
        .query('n_annotators >= 3')
        .drop('n_annotators', axis=1)
    )

    n_consiliums_for_doctor = np.zeros(N_DOCTORS, dtype=np.int32)
    tasks = []
    for doctor in range(N_DOCTORS):
        accession_numbers = nodules.query("doctor_{:03d} == 1".format(doctor)).AccessionNumber.unique()
        n_consiliums_for_doctor[doctor] = min(n_consiliums, len(accession_numbers))
        sample_accesion_numbers = np.random.choice(accession_numbers, n_consiliums_for_doctor[doctor],
                                                 replace=False)
        
        tasks.extend([(doctor, accesion_number) for accesion_number in sample_accesion_numbers])
    
    args = [
        (nodules[nodules.AccessionNumber == accession_number], doctor, factor, probabilities, confidences)
        for doctor, accession_number in tasks
    ]
    

    pool = mp.Pool()
    results = pool.map(consilium_results, args)
    pool.close()
    # results = map(consilium_results, args)
    
    new_confidences = np.zeros(N_DOCTORS)
    
    for doctor, dice in results:
        new_confidences[doctor] += dice
    new_confidences = new_confidences / n_consiliums_for_doctor

    confidences = confidences * alpha + new_confidences * (1 - alpha)
    return confidences / np.sum(confidences)
    

def consilium_results(args):
    image_nodules, doctor, factor, probabilities, confidences = args
    if image_nodules.DoctorID.isna().iloc[0]:
        return doctor, 1
    else:
        annotators = image_nodules.filter(regex='doctor_\d{3}', axis=1).sum()
        annotators = list(map(lambda x: int(x[-3:]), annotators[annotators != 0].keys()))
        annotators.remove(doctor)
        sample_annotators = np.random.choice(annotators, 2, replace=False)
        consilium_nodules = image_nodules[image_nodules.DoctorID.astype(int).isin([doctor]+list(sample_annotators))]
        mapping = {
            **{'{:03d}'.format(doctor): 0},
            **{'{:03d}'.format(value): i+1 for i, value in enumerate(sample_annotators)}
        }
        consilium_nodules.DoctorID = consilium_nodules.DoctorID.map(mapping)
        mask = create_mask(consilium_nodules, factor=factor)
            
        proba = 1 / probabilities[sample_annotators]
        proba = proba / np.sum(proba)
        proba = np.prod(proba)

        consilium_confidences = confidences[sample_annotators]
        consilium_confidences = consilium_confidences / np.sum(consilium_confidences)

        res = consilium_dice(mask, consilium_confidences)
        return doctor, res


def _compute_mask_size(nodules):
    return np.ceil(((nodules.coordX + nodules.diameter_mm + 10).max(),
                    (nodules.coordY + nodules.diameter_mm + 10).max(),
                    (nodules.coordZ + nodules.diameter_mm + 10).max())).astype(np.int32)


def _create_empty_mask(mask_size, n_doctors):
    mask_size = list(mask_size) + [n_doctors]
    res = np.zeros(mask_size)
    return res


def create_mask(consilium_nodules, factor):
    """ Create nodules mask.

    Parameters
    ----------
    nodules : pd.DataFrame

    factor : float
        ratio mm / pixels

    Returns
    -------
    mask : np.ndarray
    """
    nodules = consilium_nodules.copy()

    nodules.diameter_mm *= factor
    nodules.coordX *= factor
    nodules.coordY *= factor
    nodules.coordZ *= factor

    mask_size = list(_compute_mask_size(nodules))
    
    mask = _create_empty_mask(mask_size, 3)

    coords = np.array(nodules[['coordX', 'coordY', 'coordZ']], dtype=np.int32)
    diameters = np.array(nodules.diameter_mm, dtype=np.int32)
    values = np.array(nodules.DoctorID, dtype=np.int32)
    
    return _create_mask_numba(mask, coords, diameters, values)

@njit
def _create_mask_numba(mask, coords, diameters, values):
    for i in range(len(coords)):
        center = coords[i]
        diameter = diameters[i]
        value = values[i]

        begin_x = np.maximum(0, center[0]-diameter)
        begin_y = np.maximum(0, center[1]-diameter)
        begin_z = np.maximum(0, center[2]-diameter)

        end_x = np.minimum(mask.shape[0], center[0]+diameter+1)
        end_y = np.minimum(mask.shape[1], center[1]+diameter+1)
        end_z = np.minimum(mask.shape[2], center[2]+diameter+1)

        for x in range(begin_x, end_x):
            for y in range(begin_y, end_y):
                for z in range(begin_z, end_z):
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 < diameter ** 2:
                        mask[x, y, z, value] = 1
    return mask


def consilium_dice(mask, consilium_confidences):
    """ Compute consilium dice for current doctor.

    Parameters
    ----------
    mask : np.ndarray

    doctor : int

    consilium : list of int
        names (indices) of doctors in consilium
    confidences : np.ndarray
        confidences of all doctors

    Returns
    -------
    dice : float
        dice which is computed as dice of binary doctor mask and weighted mask of doctors by their confidences
    """
    e = 1e-6
    doctor_mask = mask[..., 0]
    consilium_mask = mask[..., 1:]
    ground_truth = np.sum(consilium_mask * consilium_confidences, axis=-1)
    # ground_truth = np.sum(consilium_mask, axis=-1) / len(consilium)

    tp = np.sum(2 * doctor_mask * ground_truth) + e
    den = np.sum(doctor_mask + ground_truth) + e

    return tp / den


def dice(mask1, mask2):
    """ Simple dice. """
    e = 1e-6

    tp = np.sum(2 * mask1 * mask2) + e
    den = np.sum(mask1 + mask2) + e

    return tp / den


def get_rating(confidences):
    """ Get list of doctors ordered by confidence. """
    return np.array([item[0] for item in sorted(enumerate(confidences), key=lambda x: -x[1])])


def get_probabilities(nodules):
    """ Get list of doctors frequencies. """
    probabilities = (
        nodules
        .drop_duplicates(subset=['AccessionNumber'])
        .set_index('AccessionNumber')
        .filter(regex=r'doctor_\d{3}', axis=1)
        .sum(axis=0)
        .transform(lambda s: s / s.sum())
    )
    return probabilities

def get_common_annotation(indices, data_path='/notebooks/data/CT/npcmr', annotation_path='/notebooks/data/CT/npcmr/ct_annotation'):
    nodules = []

    for i in ['{:02d}'.format(j) for j in indices]:
        dataset_info = (
            read_dataset_info(os.path.join(data_path, '{}/*/*/*/*/*'.format(i)), index_col=None)
            .drop_duplicates(subset=['AccessionNumber'])
            .set_index('AccessionNumber')
        )

        subset_nodules = (
            read_nodules(
                os.path.join(annotation_path, '{}_annotation.txt'.format(i)), 
                include_annotators=True, 
                drop_no_cancer=True)
            .set_index('AccessionNumber')
            .assign(coordZ=lambda df: df.loc[:, 'coordZ'] * dataset_info.loc[df.index, 'SpacingZ'],
                    coordY=lambda df: df.loc[:, 'coordY'] * dataset_info.loc[df.index, 'SpacingY'],
                    coordX=lambda df: df.loc[:, 'coordX'] * dataset_info.loc[df.index, 'SpacingX'])
        )

        subset_nodules.index = i + '_' + subset_nodules.index
        nodules.append(subset_nodules)
    
    nodules = pd.concat(nodules)
    nodules = nodules[nodules.diameter_mm < 90]
    nodules.index.name = 'AccessionNumber'
    return nodules.reset_index()
    
