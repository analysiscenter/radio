""" Functions for mapping sets of overlapping nodules into one nodule. """

import numpy as np
import pandas as pd
from scipy import stats
from numba import njit
from .parser import generate_index


@njit(cache=True)
def sphere_overlap(nodule_true, nodule_pred):
    """ Two nodules overlap volume normalized by total volume of second one.

    Parameters
    ----------
    nodule_true : ndarray
        numpy array with information about true nodule:
        nodule_true[1:] - [z,y,x] coordinates of true nodule's center,
        nodule_true[0] - diameter of true nodule.
    nodule_pred : ndarray
        numpy array with information about predicted nodule:
        nodule_pred[1:] - [z,y,x] coordinates of predicted nodule's center,
        nodule_pred[0] - diameter of predicted nodule.

    Returns
    -------
    float
        overlap volume divided by sum of input nodules' volumes.
    """
    r1, r2 = nodule_true[0] / 2, nodule_pred[0] / 2
    pos1, pos2 = nodule_true[1:], nodule_pred[1:]

    pos1_area = 4. / 3. * np.pi * r1 ** 3
    pos2_area = 4. / 3. * np.pi * r2 ** 3

    d = np.sum((pos1 - pos2) ** 2) ** 0.5

    if d >= r1 + r2:
        return 0
    elif r1 >= d + r2 or r2 >= d + r1:
        return 1.0

    volume = (np.pi * (r1 + r2 - d) ** 2
              * (d ** 2 + r1 * (2 * d - 3 * r1)
                 + r2 * (2 * d - 3 * r2)
                 + 6 * r1 * r2)) / (12 * d + 10e-7)
    return 2 * volume / (pos2_area + pos1_area + 10e-7)


@njit
def compute_overlap_distance_matrix(coords, diameters):
    """ Compute pairwise overlap matrix.

    Parameters
    ----------
    coords : ndarray(num_nodules, 3)
        cooordinates of nodules' centers.
    diameters : ndarray(num_nodules)
        diameters of nodules.

    Returns
    -------
    ndarray(num_nodules, num_nodules)
        overlap distance matrix required by clustering algorithm.
    """
    num_nodules = coords.shape[0]
    overlap_matrix = np.zeros(shape=(num_nodules, num_nodules))
    buffer = np.zeros((2, 4))
    for i in range(num_nodules):
        for j in range(num_nodules):
            buffer[0, 1:], buffer[1, 1:] = coords[i, :], coords[j, :]
            buffer[0, 0], buffer[1, 0] = diameters[i], diameters[j]
            overlap_matrix[i, j] = sphere_overlap(buffer[0, :], buffer[1, :])
    return overlap_matrix


@njit
def compute_reachable_vertices_numba(distance_matrix, vertex, threshold):
    """ Get vertices that can be reached from given vertex using distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray(num_nodules, num_nodules)
        overlap distance matrix for all nodules pairs.
    vertex : int
        input vertex.
    threshold : float
        threshold for volumetric intersection over union for pairs of nodules
        to consider them overlapping.

    Returns
    -------
    ndarray(num_vertices)
        vertices that can be reached from given vertex.
    """
    num_vertices = distance_matrix.shape[0]

    all_vertices = np.arange(num_vertices)
    reachable = np.zeros(num_vertices)
    unprocessed = np.zeros(num_vertices)
    reachable = (reachable == 1)
    unprocessed = (unprocessed == 1)
    unprocessed[vertex] = True
    while np.any(unprocessed):
        u = all_vertices[unprocessed][0]
        vertices = (distance_matrix[u, :] > threshold)

        unprocessed[np.logical_and(vertices, ~reachable)] = True

        reachable[u] = True
        unprocessed[u] = False
    return all_vertices[reachable]


@njit
def compute_clusters_numba(coords, diameters, threshold):
    """ Compute clusters for nodules represented by coordinates of centers and diameters.

    Parameters
    ----------
    coords : ndarray(num_nodules, 3)
        cooordinates of nodules' centers.
    diameters : ndarray(num_nodules)
        diameters of nodules.

    Returns
    -------
    ndarray(num_nodules)
        cluster number for each nodule.
    """
    distance_matrix = compute_overlap_distance_matrix(coords, diameters)
    num_elements = distance_matrix.shape[0]
    all_vertices = np.arange(num_elements)
    clusters = -np.ones(num_elements)
    current_cluster = 0
    while len(clusters[clusters == -1]) > 0:
        v = np.random.choice(all_vertices[clusters == -1], 1)
        cluster_vertices = compute_reachable_vertices_numba(distance_matrix, v, threshold)
        clusters[cluster_vertices] = current_cluster
        current_cluster += 1
    return clusters



def assign_nodules_group_index(nodules, threshold=0.1):
    """ Add column with name 'GroupNoduleID' containing index of group of overlapping nodules.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules locations and centers.

    threshold : float
        float from [0, 1] interval representing volumentric intersection over union.

    Returns
    -------
    pandas DataFrame
    """
    coords = nodules.loc[:, ['coordZ', 'coordY', 'coordX']].values
    diameters = nodules.loc[:, 'diameter_mm'].values
    if coords.shape[0] == 1 and diameters.shape[0]:
        return nodules.assign(GroupNoduleID=generate_index())
    clusters = np.array(compute_clusters_numba(coords, diameters, threshold), dtype=np.int)
    clusters_names = np.array([generate_index() for cluster_num in np.sort(np.unique(clusters))])
    group_nodule_id = pd.Series(clusters_names[clusters], index=nodules.index)
    return nodules.assign(GroupNoduleID=group_nodule_id)


def get_diameter_by_sigma(sigma, proba):
    """ Get diameter of nodule given sigma of normal distribution and probability of diameter coverage area.

    Transforms sigma parameter of normal distribution corresponding to cancerous nodule
    to its diameter using probability of diameter coverage area.

    Parameters
    ----------
    sigma : float
        square root of normal distribution variance.
    proba : float
        probability of diameter coverage area.

    Returns
    -------
    float
        equivalent diameter.
    """
    return 2 * sigma * stats.norm.ppf((1 + proba) / 2)  # pylint: disable=no-member


def get_sigma_by_diameter(diameter, proba):
    """ Get sigma of normal distribtuion by diameter of nodule and probability of diameter coverage area.

    Parameters
    ----------
    diameter : float
        diameter of nodule.
    proba : float
        probability of diameter coverage area.

    Returns
    -------
    float
        equivalent normal distribution's sigma parameter.
    """
    return diameter / (2 * stats.norm.ppf((1 + proba) / 2))  # pylint: disable=no-member


def approximate_gaussians(confidence_array, mean_array, variance_array):
    """ Approximate gaussians with given parameters with one gaussian.

    Approximation is performed via minimization of Kullback-Leibler
    divergence KL(sum_{j} w_j N_{mu_j, sigma_j} || N_{mu, sigma}).

    Parameters
    ----------
    confidence_array : ndarray(num_gaussians)
        confidence values for gaussians.
    mean_array : ndarray(num_gaussians, 3)
        (z,y,x) mean values for input gaussians.
    variance_array : ndarray(num_gaussians)
        (z,y,x) variances for input gaussians.

    Returns
    -------
    tuple(ndarray(3), ndarray(3))
        mean and sigma for covering gaussian.
    """
    delimiter = np.sum(confidence_array)
    mu = np.sum(mean_array.T * confidence_array, axis=1) / delimiter
    sigma = np.sqrt(np.sum((variance_array + (mean_array - mu) ** 2).T
                           * confidence_array, axis=1) / delimiter)
    return mu, sigma


def compute_group_coords_and_diameter(nodules, proba=0.8):
    """ Get coordinates of center and diameter of nodules united in group.

    For each group of overlapping nodules computes equivalent diameter and
    coordinates of center. Preserves 'confidence' and 'AccessionNumber'
    columns from source nodules dataframe. Note, that this columns
    are considered to contain same values within group.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules location and sizes.
    proba : float
        float value from [0, 1] interval. Probability of diameter coverage area
        for equivalent normal distribution.

    Returns
    -------
    pandas DataFrame
        dataframe with information about equivalent locations and diameters of
        groups of overlapping nodules.
    """
    if 'NoduleConfidence' not in nodules.columns:
        nodules = nodules.assign(NoduleConfidence=1.0)

    num_nodules = nodules.shape[0]
    confidence_array = np.zeros(num_nodules, dtype=np.float64)
    mean_array = np.zeros((num_nodules, 3), dtype=np.float64)
    variance_array = np.zeros(num_nodules, dtype=np.float64)
    for i, (_, row) in enumerate(nodules.iterrows()):
        mean_array[i, :] = np.array((row['coordZ'], row['coordY'], row['coordX']))
        variance_array[i] = get_sigma_by_diameter(row['diameter_mm'], proba=proba) ** 2
        confidence_array[i] = row['NoduleConfidence']

    variance_array = np.tile(variance_array[:, np.newaxis], (1, 3))
    approx_mean, approx_sigma = approximate_gaussians(confidence_array, mean_array, variance_array)
    return  pd.Series({'coordZ': approx_mean[0], 'coordY': approx_mean[1],
                       'coordX': approx_mean[2], 'NoduleConfidence': confidence_array.max(),
                       'AccessionNumber': nodules.AccessionNumber.iloc[0],
                       'Subset': nodules.Subset.iloc[0],
                       'diameter_mm': get_diameter_by_sigma(approx_sigma, proba=proba)[0]})


def get_nodules_groups(nodules, proba=0.8):
    """ Unite overlapping nodules in groups and compute equivalent diameter and locations of groups.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules.
    proba : float
        float from [0, 1] interval. Probability of diameter coverage area of
        equivalent normal distribution.

    Returns
    -------
    pandas DataFrame
        dataframe with information about overlapping nodules groups centers
        locations and diameters.
    """
    new_nodules = (
        nodules
        .set_index(['Subset', 'AccessionNumber'])
        .groupby(level=(0, 1))
        .apply(assign_nodules_group_index)
        .reset_index()
        .groupby('GroupNoduleID')
        .apply(compute_group_coords_and_diameter, proba=proba)
        .reset_index()
    )
    return new_nodules
