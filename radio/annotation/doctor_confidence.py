#pylint:disable=not-an-iterable
#pylint:disable=cell-var-from-loop

""" Functions to compute doctors' confidences from annotation. """

from numba import autojit, prange
import numpy as np
from tqdm import tqdm

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
    confidences_history = [confidences]
    for _ in range(n_iters):
        confidences = update_confidences(nodules, confidences, probabilities, n_consiliums, factor)
        print(confidences)
        confidences_history.append(confidences)
    return confidences_history if history else confidences

def update_confidences(nodules, confidences, probabilities, n_consiliums=10, factor=0.3, alpha=0.7):
    nodules = (
        nodules
        .assign(n_annotators=lambda df: df.filter(regex='doctor_\d{3}', axis=1).sum(axis=1))
        .query('n_annotators >= 3')
        .drop('n_annotators', axis=1)
    )

    new_confidences = []
    for doctor in range(N_DOCTORS):
        doctor_nodules = nodules.query("doctor_{:03d} == 1".format(doctor))
        accession_numbers = doctor_nodules.AccessionNumber.unique()
        sample_accesion_numbers = np.random.choice(accession_numbers, min(n_consiliums, len(accession_numbers)),
                                                 replace=False)
        res = []
        for accession_number in accession_numbers:
            image_nodules = doctor_nodules[doctor_nodules.AccessionNumber == accession_number]
            if image_nodules.DoctorID.isna().iloc[0]:
                res.append(1)
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
                
                proba = probabilities[sample_annotators]
                proba = proba / np.sum(proba)
                proba = np.prod(proba)
                
                consilium_confidences = confidences[sample_annotators]
                consilium_confidences = consilium_confidences / np.sum(consilium_confidences)
                
                res.append(proba * consilium_dice(mask, consilium_confidences))

        new_confidences.append(np.mean(res))
    new_confidences = np.array(new_confidences) / np.sum(new_confidences)
    confidences = confidences * alpha + np.array(new_confidences) * (1 - alpha)
    return confidences / np.sum(confidences)

def create_table(nodules):
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

    for i in range(N_DOCTORS):
        for j in range(i+1, N_DOCTORS):
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
                mask = create_mask(nodules, accession_number, factor=0.3)
                mask1 = mask[..., i]
                mask2 = mask[..., j]
                dices.append(dice(mask1, mask2))
            table[i, j] = np.mean(dices)
            table[j, i] = np.mean(dices)

    return table, table_meetings


def _compute_mask_size(nodules):
    return np.ceil(((nodules.coordX + nodules.diameter_mm + 10).max(),
                    (nodules.coordY + nodules.diameter_mm + 10).max(),
                    (nodules.coordZ + nodules.diameter_mm + 10).max())).astype(np.int32)

def _create_empty_mask(mask_size, n_doctors):
    mask_size = list(mask_size) + [n_doctors]
    return np.zeros(mask_size)

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

@autojit
def _create_mask_numba(mask, coords, diameters, values):
    for i, center in enumerate(coords):
        diameter = diameters[i]
        value = values[i]

        begin_x = np.maximum(0, center[0]-diameter)
        begin_y = np.maximum(0, center[1]-diameter)
        begin_z = np.maximum(0, center[2]-diameter)

        end_x = np.minimum(mask.shape[0], center[0]+diameter+1)
        end_y = np.minimum(mask.shape[1], center[1]+diameter+1)
        end_z = np.minimum(mask.shape[2], center[2]+diameter+1)

        for x in prange(begin_x, end_x):
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
        .filter(regex='doctor_\d{3}', axis=1)
        .sum(axis=0)
        .transform(lambda s: s / s.sum())
    )
    return probabilities
