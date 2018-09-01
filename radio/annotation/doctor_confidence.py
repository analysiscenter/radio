""" Functions to compute doctors' confidences from annotation. """

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
import multiprocess as mp

def get_doctors_confidences(nodules, confidences='random', n_consiliums=10, n_iters=25, n_doctors=15,
                            factor=0.3, alpha=0.7, history=False, smooth=None):
    """ Conpute confidences for doctors

    Parameters
    ----------
    nodules : pd.DataFrame
        DataFrame with columns
        `['seriesuid', 'DoctorID', 'coordZ', 'coordY', 'coordX', 'diameter_mm', 'NoduleID']`
    confidences : str or list of len n_doctors
        if 'random', initial confidences will be sampled
        if 'uniform', initial confidences is a uniform distribution
        if list, confidences of doctors
    n_consiliums : int or None
        number of consiliums for each doctor. If None, all possible consiliums will be used
    n_iters : int
        number of iterations of updating algorithm
    n_doctors : int
        number of doctors
    factor : float
        ratio for mask creation
    alpha : float
        smoothing parameter of confidence update
    history : bool
        if False, the function returns final confidences, if True, all history of updating confidences
    smooth : int or None
         if int, final confidence is a smoothed confidence by last `smooth` iterations

    Returns
    -------
    new_confidences : pd.DataFrame
    """
    annotators_info = (
        nodules
        .drop_duplicates(['seriesuid', 'DoctorID'])
        .assign(DoctorID=lambda df: ['doctor_'+str(doctor).zfill(3) for doctor in df.DoctorID])
        .pivot('seriesuid', 'DoctorID', 'DoctorID')
        .notna()
        .astype('int')
    )

    nodules = (
        nodules
        .join(annotators_info, on='seriesuid', how='left')
        .assign(n_annotators=lambda df: df.filter(regex=r'doctor_\d{3}', axis=1).sum(axis=1))
        .query('n_annotators >= 3')
        .drop('n_annotators', axis=1)
        .dropna()
    )

    for i in range(n_doctors):
        if 'doctor_{:03d}'.format(i) not in nodules.columns:
            nodules['doctor_{:03d}'.format(i)] = 0

    if confidences == 'random':
        confidences = np.ones(n_doctors) / 2 + np.random.uniform(-0.1, 0.1, n_doctors)
    elif confidences == 'uniform':
        confidences = np.ones(n_doctors) / 2

    confidences_history = [pd.DataFrame({'DoctorID': [str(i).zfill(3) for i in range(n_doctors)],
                                         'confidence': confidences, 'iteration': 0})]

    consiliums = [_consiliums_for_doctor(nodules, doctor, n_doctors) for doctor in range(n_doctors)]
    consiliums_probabilities = _consiliums_prob(nodules, n_doctors)

    for i in tqdm(range(n_iters)):
        confidences = _update_confidences(nodules, confidences, consiliums, n_consiliums,
                                          consiliums_probabilities, n_doctors, factor, alpha)
        confidences_history.append(pd.DataFrame({'DoctorID': [str(i).zfill(3) for i in range(n_doctors)],
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

def _consiliums_prob(nodules, n_doctors):
    result = np.zeros((n_doctors, n_doctors))
    denom = 0
    for i, j in list(zip(*np.triu_indices(n_doctors, k=1))):
        doc1, doc2 = ['doctor_{:03d}'.format(doctor) for doctor in [i, j]] #pylint:disable=cell-var-from-loop
        accession_numbers = (
            nodules[['seriesuid', doc1, doc2]]
            .assign(together=lambda df: df[doc1] == df[doc2]) #pylint:disable=cell-var-from-loop
            .drop_duplicates()
        )
        accession_numbers = accession_numbers[accession_numbers[doc1] == 1]
        accession_numbers = accession_numbers[accession_numbers.together]
        result[i, j] = result[j, i] = len(accession_numbers)
        denom += len(accession_numbers)
    return result / denom

def _consiliums_for_doctor(nodules, doctor, n_doctors):
    id_and_consiliums = []
    for seriesuid in nodules.query("doctor_{:03d} == 1".format(doctor)).seriesuid.unique():
        image_nodules = nodules[nodules.seriesuid == seriesuid]
        annotators = image_nodules.filter(regex=r'doctor_\d{3}', axis=1).sum()
        annotators = [int(name[-3:]) for name in annotators[annotators != 0].keys()]
        annotators.remove(doctor)
        consiliums = itertools.combinations(annotators, 2)
        id_and_consiliums.extend(list(itertools.product([seriesuid], consiliums)))
    return id_and_consiliums

def _update_confidences(nodules, confidences, consiliums, n_consiliums, consiliums_probabilities,
                        n_doctors=15, factor=0.3, alpha=0.7):
    args = []
    for doctor, doctor_consiliums in enumerate(consiliums):
        if len(doctor_consiliums) != 0:
            if n_consiliums is None:
                sample = doctor_consiliums
            else:
                sample_indices = np.random.choice(
                    len(doctor_consiliums),
                    size=min(n_consiliums, len(doctor_consiliums)),
                    replace=False
                )
                sample = [doctor_consiliums[i] for i in sample_indices]
            for seriesuid, consilium in sample:
                args.append((nodules[nodules.seriesuid == seriesuid], doctor,
                             consilium, factor, confidences))

    pool = mp.Pool() #pylint:disable=no-member
    results = pool.map(_consilium_results, args)
    pool.close()

    new_confidences = confidences.copy()
    sum_weights = np.zeros(n_doctors)

    for doctor, consilium, score in results:
        weight = 1 / consiliums_probabilities[consilium[0], consilium[1]]
        new_confidences[doctor] += weight * score
        sum_weights[doctor] += weight
    new_confidences = new_confidences / sum_weights

    confidences = confidences * alpha + new_confidences * (1 - alpha)
    return confidences


def _consilium_results(args):
    image_nodules, doctor, consilium, factor, confidences = args
    if image_nodules.DoctorID.isna().iloc[0]:
        return doctor, 1
    else:
        mask = create_mask(image_nodules, doctor, consilium, factor=factor)
        consilium_confidences = confidences[list(consilium)]
        consilium_confidences = consilium_confidences / np.sum(consilium_confidences)
        print(consilium_confidences)
        return doctor, consilium, consilium_dice(mask, consilium_confidences)


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

    nodules.coordX -= nodules.coordX.min() - nodules.diameter_mm.max()
    nodules.coordY -= nodules.coordY.min() - nodules.diameter_mm.max()
    nodules.coordZ -= nodules.coordZ.min() - nodules.diameter_mm.max()

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
    for i in range(len(coords)): #pylint:disable=consider-using-enumerate
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
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 < (diameter) ** 2:
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
    doctor_mask = mask[..., 0]
    consilium_mask = mask[..., 1:]
    consilium_confidences = np.array([0.5, 0.5])
    ground_truth = np.sum(consilium_mask * consilium_confidences, axis=-1)

    return dice(doctor_mask, ground_truth)


def dice(mask1, mask2):
    """ Simple dice. """
    e = 1e-6

    tp = np.sum(2 * mask1 * mask2) + e
    den = np.sum(mask1 + mask2) + e

    return tp / den


def get_rating(confidences):
    """ Get list of doctors ordered by confidence. """
    return np.array([item[0] for item in sorted(enumerate(confidences), key=lambda x: -x[1])])


def get_table(nodules, n_doctors=15, factor=0.3):
    """ Create tables.

    Parameters
    ----------
    nodules : pd.DataFrame

    n_doctors : int
        number of doctors
    factor : float
        ratio for mask creation

    Returns
    -------
    np.ndarray
        table of the mean dice between two doctors
    np.ndarray
        table of the number of meetings between two doctors
    """
    table = np.zeros((n_doctors, n_doctors))
    table_meetings = np.zeros((n_doctors, n_doctors))

    for i, j in tqdm(list(zip(*np.triu_indices(n_doctors, k=1)))):
        accession_numbers = (
            nodules
            .groupby('seriesuid')
            .apply(lambda x: i in x.DoctorID.astype(int).values and j in x.DoctorID.astype(int).values) #pylint:disable=cell-var-from-loop
        )
        accession_numbers = accession_numbers[accession_numbers].index
        table_meetings[i, j] = len(accession_numbers)
        table_meetings[j, i] = len(accession_numbers)
        dices = []
        for accession_number in accession_numbers:
            if len(nodules[nodules.seriesuid == accession_number].dropna()) != 0:
                mask = create_mask(nodules[nodules.seriesuid == accession_number].dropna(), i, [j], factor)
                mask1 = mask[..., 0]
                mask2 = mask[..., 1]
                dices.append(dice(mask1, mask2))
            else:
                dices.append(1)
        table[i, j] = table[j, i] = np.mean(dices)

    return table, table_meetings

def generate_nodule(size=1):
    """ Generate nodules

    Parameters
    ----------
    size : int
        number of nodules

    Returns
    -------
    dict
        dict of nodules with keys `coordX, coordY, coordZ, diameter_mm`
    """
    coordX = np.random.uniform(-230.9075 / 2, 1191.48908 / 2, size)
    coordY = np.random.uniform(-229.933 / 2, 1356.4289999999999 /2, size)
    coordZ = np.random.uniform(-1147.5 / 2, 4528.5 / 2, size)
    diameter_mm = np.random.randint(1, 15, size)
    return {'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ, 'diameter_mm': diameter_mm}

def generate_annotation(n_images, n_doctors=10, bad_doctors=None, middle_doctors=None):
    """ Generate annotation

    Parameters
    ----------
    n_images : int

    n_doctors : int

    bad_doctors : list or None (then `[0]`)
        indices of doctors who select nodules that don't coincide with nodules of other doctors
    middle_doctors : list or None (then `[]`)
        indices of doctors who select nodules that don't coincide with nodules of other doctors
        and also select nodules that coincide with nodules of good doctors.
        All good doctors always select the same nodules.

    Returns
    -------
    pd.DataFrame
        annotation with columns `['seriesuid', 'DoctorID', 'coordZ', 'coordY', 'coordX', 'diameter_mm']`
    """
    bad_doctors = bad_doctors or [0]
    middle_doctors = middle_doctors or [0]
    annotation = pd.DataFrame({'seriesuid': [], 'DoctorID': [], 'coordX': [],
                               'coordY': [], 'coordZ': [], 'diameter_mm': []})
    for i in range(n_images):
        doctors = np.random.choice(np.arange(0, n_doctors), size=np.random.randint(3, 5), replace=False)

        for doctor in bad_doctors + middle_doctors:
            if doctor in doctors:
                doctor_find = np.random.randint(0, 4)
                doctor_nodules = pd.DataFrame({
                    'seriesuid': [str(i)] * doctor_find,
                    'DoctorID': [str(doctor).zfill(3)] * doctor_find,
                    **generate_nodule(doctor_find)
                })
                annotation = pd.concat([annotation, doctor_nodules], axis=0)            # pylint: disable=redefined-variable-type

        consilium_find = np.random.randint(0, 4)
        nod = generate_nodule(consilium_find)

        for d in doctors:
            if d not in bad_doctors:
                doctor_nodules = pd.DataFrame({
                    'seriesuid': [str(i)] * consilium_find,
                    'DoctorID': [str(d).zfill(3)] * consilium_find,
                    **nod
                })
            annotation = pd.concat([annotation, doctor_nodules], axis=0)
    annotation = annotation.reset_index()
    del annotation['index']
    return annotation
