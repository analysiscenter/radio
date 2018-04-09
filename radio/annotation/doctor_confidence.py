""" Functions to compute doctors' confidences from annotation. """

from numba import njit, autojit, prange
import numpy as np
import pandas as pd
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
    for j in range(n_iters):
        confidences = update_confidences(nodules, confidences, probabilities, n_consiliums, factor)
        confidences_history.append(confidences)
    return confidences_history if history else confidences

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
            accession_numbers = (nodules
                                 .groupby('AccessionNumber')
                                 .apply(lambda x: i in x.DoctorID.astype(int).values and j in x.DoctorID.astype(int).values)
                                )
            accession_numbers = accession_numbers[accession_numbers == True].index
            table_meetings[i, j] = len(accession_numbers)
            table_meetings[j, i] = len(accession_numbers)
            dices = []
            for accession_number in accession_numbers:
                mask = create_mask(nodules, accession_number, factor=0.3)
                mask1 = mask[..., i]
                mask2 = mask[..., j]
                dices.append(dice(mask1, mask2))
            table[i,j] = np.mean(dices)
            table[j,i] = np.mean(dices)

    return table, table_meetings

def update_confidence(nodules, doctor, confidences, probabilities, n_consiliums=3, factor=0.3):
    """ Update doctor confidence
    
    Parameters
    ----------
    nodules : pd.DataFrame

    doctor : int

    confidences : np.ndarray
        initial confidences to update
    probabilities : np.ndarray
        frequences of doctors in studies
    n_consiliums : int
        number of consiliums for each doctor
    factor : float
        ratio for mask creation
    
    Returns
    -------
    new_confidence : int
    """
    accession_numbers = nodules[nodules.DoctorID.astype(int) == doctor].AccessionNumber.unique()
    accession_numbers = np.random.choice(accession_numbers,
                                         min(n_consiliums, len(accession_numbers)),
                                         replace=False)
    res = []
    for accession_number in accession_numbers:
        consilium = nodules[nodules.AccessionNumber == accession_number][['annotator_1', 'annotator_2', 'annotator_3']].astype(int).iloc[0]
        consilium = np.array(consilium)
        consilium = np.delete(consilium, np.argwhere(consilium == doctor))
        mask = create_mask(nodules, accession_number, factor=factor)
        # mask = load_mask(accession_number, 8)
        proba = np.prod(probabilities[consilium])
        proba = proba / np.sum(proba)
        res.append(proba * consilium_dice(mask, doctor, consilium, confidences))
    return np.mean(res)

def update_confidences(nodules, confidences, probabilities, n_consiliums=10, factor=0.3):
    """ Update all confidences
    
    Parameters
    ----------
    nodules : pd.DataFrame

    doctor : int

    confidences : np.ndarray
        initial confidences to update
    probabilities : np.ndarray
        frequences of doctors in studies
    n_consiliums : int
        number of consiliums for each doctor
    factor : float
        ratio for mask creation
    
    Returns
    -------
    new_confidences : np.ndarray
    """
    alpha = 0.7
    new_confidences = [update_confidence(nodules, i, confidences, probabilities, n_consiliums, factor) for i in np.arange(N_DOCTORS)]
    new_confidences = np.array(new_confidences) / np.sum(new_confidences)
    confidences = confidences * alpha + np.array(new_confidences) * (1 - alpha)
    return confidences / np.sum(confidences)

def _compute_mask_size(nodules):
    return np.ceil(((nodules.coordX + nodules.diameter_mm + 10).max(),
                    (nodules.coordY + nodules.diameter_mm + 10).max(),
                    (nodules.coordZ + nodules.diameter_mm + 10).max())).astype(np.int32)
    
def _create_empty_mask(mask_size, doctors, n_doctors):
    mask_size = list(mask_size) + [n_doctors]
    mask = np.ones(mask_size) * (-1)
    mask[:, :, :, doctors] = 0
    return mask

def create_mask(nodules, accession_number, factor=1.):
    """ Create nodules mask.

    Parameters
    ----------
    nodules : pd.DataFrame

    accession_number : str

    factor : float
        ratio mm / pixels
    
    Returns
    -------
    mask : np.ndarray
    """
    n_doctors = len(nodules.DoctorID.unique())
    
    nodules = nodules.copy()
    
    nodules.diameter_mm *= factor
    nodules.coordX *= factor
    nodules.coordY *= factor
    nodules.coordZ *= factor
    
    image_nodules = nodules[nodules.AccessionNumber == accession_number]
    mask_size = list(_compute_mask_size(image_nodules))
    values = np.array(image_nodules.DoctorID.astype(int), dtype=np.int32)
    
    doctors = np.array(image_nodules[['annotator_1', 'annotator_2', 'annotator_3']].astype(int).iloc[0])
    
    mask = _create_empty_mask(mask_size, doctors, n_doctors)
    coords = np.array(image_nodules[['coordX', 'coordY', 'coordZ']], dtype=np.int32)
    diameters = np.array(image_nodules.diameter_mm, dtype=np.int32)
    
    return _create_mask_numba(mask, coords, diameters, values)

def save_masks(nodules, folder='masks', factor=1.):
    """ Save all masks

    Parameters
    ----------
    nodules : pd.DataFrame

    folder : str
        path to save masks
    factor : float
        ratio mm / pixels
    """
    for accession_number in tqdm(nodules.AccessionNumber.unique()):
        mask = create_mask(nodules, accession_number, factor)
        with open('{}/{}'.format(folder, accession_number), 'wb') as file:
            np.save(file, mask)

def load_mask(accession_number, folder='masks'):
    """ Load mask

    Parameters
    ----------
    accession_number : str

    folder : str
        path where masks were saved
    """
    with open('{}/{}'.format(folder, accession_number), 'rb') as file:
        mask = np.load(file)
    return mask
            
@autojit
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
        
        for x in prange(begin_x, end_x):
            for y in range(begin_y, end_y):
                for z in range(begin_z, end_z):
                    if ( (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 < diameter ** 2):
                        mask[x, y, z, value] = 1
    return mask

def consilium_dice(mask, doctor, consilium, confidences):
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
    doctor_mask = mask[..., doctor]
    consilium_mask = mask[..., consilium]
    consilium_confidences = confidences[consilium]
    consilium_confidences = consilium_confidences / np.sum(consilium_confidences)
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
    probabilities = (nodules
        .groupby('AccessionNumber')
        .apply(lambda x: np.array([i in x.DoctorID.unique().astype(int) for i in range(N_DOCTORS)], dtype=np.int32))
        .sum())
    return probabilities / np.sum(probabilities)