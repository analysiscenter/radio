#pylint:disable=not-an-iterable
#pylint:disable=cell-var-from-loop
#pylint:disable=consider-using-enumerate
#pylint:disable=redefined-variable-type

""" Functions to compute doctors' confidences from annotation. """

import os
import multiprocessing as mp
from numba import njit
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import read_nodules, read_dataset_info, read_annotators_info

N_DOCTORS = 15


def get_doctors_confidences(nodules, confidences='random', n_iters=25, n_consiliums=10,
                            factor=0.3, alpha=0.7, history=False, smooth=None):
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
    alpha : float
        smoothing parameter of confidence update
    history : bool
        if False, the function returns final confidences, if True, all history of updating confidences

    Returns
    -------
    new_confidences : pd.DataFrame
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
        confidences = _update_confidences(nodules, confidences, probabilities, n_consiliums, factor, alpha)
        confidences_history.append(pd.DataFrame({'DoctorID': [str(i).zfill(3) for i in range(N_DOCTORS)],
                                                 'confidence': confidences, 'iteration': i+1}))
    if history:
        res = pd.concat(confidences_history, axis=0)
    else:
        if smooth is None:
            res = confidences_history[-1].drop(columns=['iteration'])
        else:
            res = (
                pd
                .concat(confidences_history, axis=0)
                .pivot(index='iteration', columns='DoctorID', values='confidence')
            )
            res = (
                nodules
                .set_index('DoctorID')
                .assign(DoctorConfidence=res.rolling(smooth).mean().iloc[-1, :])
                .reset_index()
            )
    return res


def _update_confidences(nodules, confidences, probabilities, n_consiliums=10, factor=0.3, alpha=0.7):
    nodules = (
        nodules
        .assign(n_annotators=lambda df: df.filter(regex=r'doctor_\d{3}', axis=1).sum(axis=1))
        .query('n_annotators >= 3')
        .drop('n_annotators', axis=1)
    )

    tasks = []
    for doctor in range(N_DOCTORS):
        accession_numbers = nodules.query("doctor_{:03d} == 1".format(doctor)).AccessionNumber.unique()
        sample_accesion_numbers = np.random.choice(accession_numbers, min(n_consiliums, len(accession_numbers)),
                                                   replace=False)

        tasks.extend([(doctor, accesion_number) for accesion_number in sample_accesion_numbers])

    args = [
        (nodules[nodules.AccessionNumber == accession_number], doctor, factor, confidences)
        for doctor, accession_number in tasks
    ]


    pool = mp.Pool()
    results = pool.map(_consilium_results, args)
    pool.close()

    new_confidences = np.zeros(N_DOCTORS)
    sum_weights = np.zeros(N_DOCTORS)

    for doctor, annotators, score in results:
        weight = np.prod(1 / probabilities[annotators]) / probabilities[doctor]
        new_confidences[doctor] += weight * score
        sum_weights[doctor] += weight
    new_confidences = new_confidences / sum_weights

    confidences = confidences * alpha + new_confidences * (1 - alpha)
    return confidences / np.sum(confidences)


def _consilium_results(args):
    image_nodules, doctor, factor, confidences = args
    if image_nodules.DoctorID.isna().iloc[0]:
        return doctor, 1
    else:
        annotators = image_nodules.filter(regex=r'doctor_\d{3}', axis=1).sum()
        annotators = [int(name[-3:]) for name in annotators[annotators != 0].keys()]
        annotators.remove(doctor)
        sample_annotators = np.random.choice(annotators, 2, replace=False)
        mask = create_mask(image_nodules, doctor, sample_annotators, factor=factor)

        consilium_confidences = confidences[sample_annotators]
        consilium_confidences = consilium_confidences / np.sum(consilium_confidences)

        return doctor, annotators, consilium_dice(mask, consilium_confidences)


def _compute_mask_size(nodules):
    return np.ceil(((nodules.coordX + nodules.diameter_mm + 10).max(),
                    (nodules.coordY + nodules.diameter_mm + 10).max(),
                    (nodules.coordZ + nodules.diameter_mm + 10).max())).astype(np.int32)


def _create_empty_mask(mask_size, n_doctors):
    mask_size = list(mask_size) + [n_doctors]
    res = np.zeros(mask_size)
    return res


def create_mask(image_nodules, doctor, annotators, factor):
    """ Create nodules mask.

    Parameters
    ----------
    image_nodules : pd.DataFrame

    doctor : int
        doctor to estimate
    annotators : list or np.array
        doctors in consilium
    factor : float
        ratio mm / pixels

    Returns
    -------
    mask : np.ndarray
    """
    nodules = image_nodules.copy()

    nodules.diameter_mm *= factor
    nodules.coordX *= factor
    nodules.coordY *= factor
    nodules.coordZ *= factor

    mask_size = list(_compute_mask_size(nodules))

    mask = _create_empty_mask(mask_size, len(annotators)+1)

    for i, annotator in enumerate([doctor] + list(annotators)):
        annotator_nodules = nodules[nodules.DoctorID.astype(int) == annotator]
        coords = np.array(annotator_nodules[['coordX', 'coordY', 'coordZ']], dtype=np.int32)
        diameters = np.array(annotator_nodules.diameter_mm, dtype=np.int32)
        mask[..., i] = _create_mask_numba(mask[..., i], coords, diameters)

    return mask

@njit
def _create_mask_numba(mask, coords, diameters):
    for i in range(len(coords)):
        center = coords[i]
        diameter = diameters[i]

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
                        mask[x, y, z] = 1
    return mask


def consilium_dice(mask, consilium_confidences):
    """ Compute consilium dice for current doctor.

    Parameters
    ----------
    mask : np.ndarray

    consilium_confidences : np.ndarray

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

def get_common_annotation(indices, data_path, annotation_path):
    """ Get annotations with all coordinates in mm

    Parameters
    ----------
    indices : list
        indices od subsets to load
    data_path : str
        path to folder with data
    annotation_path : str
        path to folder with annotations

    Returns
    -------
    pd.DataFrame
    """
    nodules = []

    for i in tqdm(['{:02d}'.format(j) for j in indices]):
        annotation = os.path.join(annotation_path, '{}_annotation.txt'.format(i))
        annotators = read_annotators_info(annotation, annotator_prefix='doctor_')

        dataset_info = (
            read_dataset_info(os.path.join(data_path, '{}/*/*/*/*/*'.format(i)), index_col=None)
            .drop_duplicates(subset=['AccessionNumber'])
            .set_index('AccessionNumber')
        )

        subset_nodules = (
            read_nodules(annotation)
            .set_index('AccessionNumber')
            .assign(coordZ=lambda df: df.loc[:, 'coordZ'] * dataset_info.loc[df.index, 'SpacingZ'],
                    coordY=lambda df: df.loc[:, 'coordY'] * dataset_info.loc[df.index, 'SpacingY'],
                    coordX=lambda df: df.loc[:, 'coordX'] * dataset_info.loc[df.index, 'SpacingX'])
            .merge(annotators, left_index=True, right_index=True)
        )

        subset_nodules.index = i + '_' + subset_nodules.index
        nodules.append(subset_nodules)

    nodules = pd.concat(nodules)
    nodules = nodules[nodules.diameter_mm < 90]
    nodules.index.name = 'AccessionNumber'
    return nodules.reset_index()

def get_table(nodules, factor=0.3):
    """ Create tables.
    Parameters
    ----------
    nodules : pd.DataFrame
    Returns
    -------
    table : np.ndarray
        table of the mean dice between two doctors
    table : np.ndarray
        table of the number of meetings between two doctors
    """
    table = np.zeros((N_DOCTORS, N_DOCTORS))
    table_meetings = np.zeros((N_DOCTORS, N_DOCTORS))

    for i, j in tqdm(list(zip(*np.triu_indices(N_DOCTORS, k=1)))):
        accession_numbers = (
            nodules
            .groupby('AccessionNumber')
            .apply(lambda x: i in x.DoctorID.astype(int).values and j in x.DoctorID.astype(int).values)
        )
        accession_numbers = accession_numbers[accession_numbers].index
        table_meetings[i, j] = len(accession_numbers)
        table_meetings[j, i] = len(accession_numbers)
        dices = []
        for accession_number in accession_numbers:
            if len(nodules[nodules.AccessionNumber == accession_number]) != 0:
                try:
                    mask = create_mask(nodules[nodules.AccessionNumber == accession_number], i, [j], factor)
                except:
                    raise Exception(nodules[nodules.AccessionNumber == accession_number], i, j)
                mask1 = mask[..., 0]
                mask2 = mask[..., 1]
                dices.append(dice(mask1, mask2))
            else:
                dices.append(1)
        table[i, j] = np.mean(dices)
        table[j, i] = np.mean(dices)

    return table, table_meetings
