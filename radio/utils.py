""" Contains functions that can be helpful for initial CT-dataset manipulations. """
import os
import sys
from functools import partial
from binascii import hexlify
from collections import OrderedDict
import pickle
import glob
import numpy as np
import scipy.stats as stats
import pandas as pd
import dicom


from .models.utils import sphere_overlap
from .pipelines import get_crops
from .preprocessing import CTImagesMaskedBatch
from .dataset import F, FilesIndex, Dataset


def generate_index(size=20):
    """ Generate random string index of givne size.

    Parameters
    ----------
    size : int
        length of index string.

    Returns
    -------
        string index of given size.
    """
    return hexlify(np.random.rand(100))[:size].decode()


def normalize_nodule_type(nodules):
    """ Normalize info contained in 'NoduleType' column of dataframe with nodule info.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with info about nodules' locations.

    Returns
    -------
    pandas DataFrame
        copy of input dataframe with normalized 'NoduleType' column.
    """
    nodule_type = nodules.loc[:, 'NoduleType']
    nodule_type = nodule_type.str.strip()
    nodule_type = nodule_type.str.lower()

    nodule_type = nodule_type.str.replace('каверна', 'c')
    nodule_type = nodule_type.str.replace('nan', 'c')

    nodule_type = nodule_type.str.replace('g|ч|x', 'п')
    nodule_type = nodule_type.str.replace('а|a|k|к|‘', 'с')
    nodule_type = nodule_type.str.replace('v', 'м')
    nodule_type = nodule_type.str.replace('^$', 'c')

    nodule_type = nodule_type.str.replace('(?:n|п)(?:c|с)?', 'semi_solid')
    nodule_type = nodule_type.str.replace('м', 'ground_glass')
    nodule_type = nodule_type.str.replace('(?:c|с)(\d|\s.+)?', 'solid')

    return nodules.assign(NoduleType=nodule_type)


def get_dicom_info(paths, index_col=None):
    """ Accumulates scans meta information from dicom dataset in pandas DataFrame.

    Parameters
    ----------
    paths : list, tuple or ndarray of strings
        paths to directories with dicom files.
    index_col : str or None
        name of column that will be used as index of output DataFrame.

    Returns
    -------
    DataFrame
        pandas DataFrame with scans' meta information.
    """
    meta_info = []
    indices = []
    for path in paths:
        first_slice = dicom.read_file(os.path.join(path, os.listdir(path)[0]))

        info_dict = {
            'SpacingZ': float(first_slice.SliceThickness),
            'SpacingY': float(first_slice.PixelSpacing[0]),
            'SpacingX': float(first_slice.PixelSpacing[1]),
            'StudyID': str(first_slice.StudyID),
            'AccessionNumber': str(first_slice.AccessionNumber),
            'PatientID': str(first_slice.PatientID),
            'Rows': int(first_slice.Rows),
            'Columns': int(first_slice.Columns),
            'NumSlices': len(os.listdir(path)),
            'ScanID': os.path.basename(path),
            'Index': str(first_slice.AccessionNumber) + '_' + os.path.basename(path),
            'ScanPath': path
        }
        meta_info.append(info_dict)
    return pd.DataFrame(meta_info) if index_col is None else pd.DataFrame(meta_info).set_index(index_col)


def filter_dicom_info_by_best_spacing(info_df):
    """ Filter dataframe created by get_dicom_info function by minimal z-spacing.

    This function groups by items in dataframe created by get_dicom_info function
    and after that takes only rows corresponding to minimal z-spacing value
    for given AccessionNumber.

    Parameters
    ----------
    info_df : pandas DataFrame
        descriptive information for dicom dataset. Commonly the output of
        get_dicom_info function.

    Returns
    -------
        pandas DataFrame
    """
    output_indices = (
        info_df
        .groupby('AccessionNumber')
        .agg({'SpacingZ': 'idxmin'})
    )
    info_df = info_df.loc[output_indices.loc[:, 'SpacingZ'], :]
    return info_df


def parse_annotation(path, max_nodules=40):
    """ Parse annotation provided by doctors.

    Parameters
    ----------
    path : str
        path to file with annotation.
    max_nodules : str
        maximum number of cancer nodules found in patient.

    Returns
    -------
        pandas DataFrame with annotation.
    """
    base_columns = ['AccessionNumber', 'StudyID', 'DoctorID', 'Comment', 'NumNodules']
    location_columns = ['locX', 'locY', 'locZ', 'diam', 'type']

    nodules_list = []

    with open(path, encoding='utf-16') as file:
        data = (
            file
            .read()
            .replace('Оценки эксперта', '')
            .split('\n\n\n')[1:]
        )
        for part in data:
            for estimate in part.split('\n'):
                estimate_dict = OrderedDict()
                values = estimate.split('\t')
                for i in range((max_nodules + 1) * 5):
                    value = values[i] if i < len(values) else 'NaN'
                    if i <= 4:
                        estimate_dict[base_columns[i]] = value
                    else:
                        preffix = location_columns[(i - 4) % 5 - 1]
                        suffix = str((i - 5) // 5)
                        estimate_dict[preffix + '_' + suffix] = 'NaN' if value == '-' else value
                nodules_list.append(estimate_dict)

    return pd.DataFrame(nodules_list)

def annotation_to_nodules(annotation_df):
    """ Transform dataframe with annotation to dataframe with information about nodules.

    Each row in the output dataframe corresponds to separate nodule.

    Parameters
    ----------
    annotation_df : pandas DataFrame
        pandas dataframe with annotation, usually output of 'parse_annotation'
        function.

    Returns
    -------
    pandas DataFrame
    """
    data_list = []
    for group in annotation_df.groupby(['AccessionNumber', 'DoctorID']):
        accession_number = group[0][0]
        doctor_id = group[0][1]

        nodules = group[1].iloc[:, 5:].values.reshape(-1, 5)
        for i in range(nodules.shape[0]):
            nodule_id = generate_index()
            nodule_dict = {
                'AccessionNumber': accession_number,
                'DoctorID': doctor_id,
                'NoduleID': nodule_id,
                'NoduleType': nodules[i, 4],
                'coordX': nodules[i, 0] if nodules[i, 0] != '' else 'NaN',
                'coordY': nodules[i, 1] if nodules[i, 1] != '' else 'NaN',
                'coordZ': nodules[i, 2] if nodules[i, 2] != '' else 'NaN',
                'diameter_mm': nodules[i, 3] if nodules[i, 3] != '' else 'NaN',
            }
            data_list.append(nodule_dict)
    result_df = pd.DataFrame(data_list)
    result_df.coordX = result_df.coordX.astype(np.float)
    result_df.coordY = result_df.coordY.astype(np.float)
    result_df.coordZ = result_df.coordZ.astype(np.float)
    result_df.diameter_mm = result_df.diameter_mm.astype(np.float)
    result_df = result_df.dropna()
    result_df = result_df.assign(DoctorID=lambda df: df.loc[:, 'DoctorID'].str.replace("'", ""))
    return normalize_nodule_type(result_df)


def assign_nodules_group_index(nodules):
    """ Add column with name 'GroupNoduleID' containing index of group of overlapping nodules.

    Parameters
    ----------
    nodules : pandas DataFrame
        dataframe with information about nodules locations and centers.

    Returns
    -------
    pandas DataFrame
    """
    overlap_groups = {}
    for nodule_l, row_l in nodules.iterrows():
        overlap_indices = []
        for nodule_r, row_r in nodules.iterrows():
            al = row_l.loc[['diameter_mm', 'coordZ', 'coordY', 'coordX']].values.astype(np.float)
            ar = row_r.loc[['diameter_mm', 'coordZ', 'coordY', 'coordX']].values.astype(np.float)

            if sphere_overlap(al, ar) > 0:
                overlap_indices.append(nodule_r)

        if not any(nodule_id in overlap_groups for nodule_id in overlap_indices):
            index = generate_index()
        else:
            nodules_list = [nodule_id for nodule_id in overlap_indices if nodule_id in overlap_groups]
            index = overlap_groups[nodules_list[0]]

        for nodule_id in overlap_indices:
            overlap_groups[nodule_id] = index

    return nodules.assign(GroupNoduleID=pd.Series(overlap_groups))


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
    return 2 * sigma * stats.norm.ppf((1 + proba) / 2)


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
    return diameter / (2 * stats.norm.ppf((1 + proba) / 2))


def approximate_gaussians(confidence_array, mean_array, variance_array):
    """ Approximate gaussians with given parameters with one gaussian.

    Approximation is performed via minimization of Kullback-Leibler
    divergence KL(\sum_{j} w_j N_{\mu_j, \sigma_j} || N_{\mu, \sigma}).

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
    num_nodules = nodules.shape[0]
    confidence_array = np.zeros(num_nodules, dtype=np.float64)
    mean_array = np.zeros((num_nodules, 3), dtype=np.float64)
    variance_array = np.zeros(num_nodules, dtype=np.float64)
    for i, (nodule_id, row) in enumerate(nodules.iterrows()):
        mean_array[i, :] = np.array((row['coordZ'], row['coordY'], row['coordX']))
        variance_array[i] = get_sigma_by_diameter(row['diameter_mm'], proba=proba) ** 2
        confidence_array[i] = row['confidence']

    variance_array = np.tile(variance_array[:, np.newaxis], (1, 3))
    approx_mean, approx_sigma = approximate_gaussians(confidence_array, mean_array, variance_array)
    return  pd.Series({'coordZ': approx_mean[0], 'coordY': approx_mean[1],
                       'coordX': approx_mean[2], 'confidence': confidence_array.max(),
                       'AccessionNumber': nodules.AccessionNumber.iloc[0],
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
        .set_index(['AccessionNumber', 'NoduleID'])
        .groupby(level=0)
        .apply(assign_nodules_group_index)
        .reset_index()
        .groupby('GroupNoduleID')
        .apply(compute_group_coords_and_diameter, proba=proba)
        .reset_index()
    )
    return new_nodules


def read_nodules(paths, include_annotators=False, drop_no_cancer=False):
    """ Read annotation from file and transform it to dataframe with nodules.

    Parameters
    ----------
    path : str
        path to file with annotation.
    include_annotators : bool
        whether to include binary columns about annotators.
    drop_no_cancer : bool
        whether to drop rows without nodules.

    Returns
    -------
    pandas DataFrame
        dataframe that contains information about nodules location, type and etc.
    """
    if type(paths) is str: paths = [paths]
    out_nods = []
    for path in paths:
        annotation = parse_annotation(path)
        annotators_info = (
            annotation
            .assign(DoctorID=lambda df: df.DoctorID.str.replace("'", ""))
            .query("AccessionNumber != ''")
            .pivot('AccessionNumber', 'DoctorID', 'DoctorID')
            .notna()
            .astype('int')
            .pipe(lambda df: df.rename(columns={doctor_id: 'doctor_' + str(doctor_id)
                                                for doctor_id in df.columns}))
        )
        nodules = annotation_to_nodules(annotation)

        if include_annotators:
            nodules = pd.merge(annotation_to_nodules(annotation), annotators_info,
                               left_on='AccessionNumber', right_index=True,
                               how='inner' if drop_no_cancer else 'right')
        out_nods += [nodules]
    nodules = pd.concat(out_nods)
    return nodules


def read_dataset_info(path=None, paths=None, index_col=None, filter_by_min_spacing=False):
    """ Build index and mapping to paths for given dicom dataset.

    Parameters
    ----------
    path : str
        dataset scans path mask that will be used as glob.glob argument.
        Default is None. Only one of 'path' and 'paths' arguments must be not None.
    paths : list
        list of scans paths. Default is None. Only one of 'path' and 'paths'
        arguments must be not None.
    index_col : str or None.
        name of column in the output dataframe that will be used as index.
        Default is None.

    Returns
    -------
    pandas DataFrame
        dataframe containing info about dicom dataset.
    """
    if (path is None and paths is None) or (path is not None and paths is not None):
        raise ValueError("Only one of 'path' or 'paths' arguments must be provided")

    dataset_info = get_dicom_info(glob.glob(path) if path is not None else paths)
    if filter_by_min_spacing:
        output_indices = (
            dataset_info
            .groupby('AccessionNumber')
            .agg({'SpacingZ': 'idxmin'})
        )
        index_df = dataset_info.loc[output_indices.loc[:, 'SpacingZ'], :]
    else:
        index_df = dataset_info
    return index_df if index_col is None else index_df.set_index(index_col)


def get_dicom_dataset_and_nodules(dataset_path, nodules=None):
    """ Get dicom dataset and nodules given dataset and nodules DataFrame created using annotation file.

    Parameters
    ----------
    dataset_path : str
        path to directory where dicom dataset is located.
    nodules : pandas DataFrame
        dataframe obtained from read_nodules function or None. Default value is None.

    Returns
    -------
    tuple(dataset.Dataset, pandas.DataFrame)
        Dataset object from dataset submodule and source nodules pandas DataFrame
        with seriesuid column added.
    """
    index_df = read_dataset_info(os.path.join(dataset_path, "/*/*/*/*/*"))
    raw_indices = index_df.index.values
    mapping = dict(index_df.loc[:, 'ScanPath'])

    index = FilesIndex(raw_indices, paths=mapping, dirs=True)
    dataset = Dataset(index, batch_class=CTImagesMaskedBatch)

    if nodules is not None:
        nodules_mapping = (
            index_df
            .reset_index()
            .loc[:, ['AccessionNumber', 'Index']]
        )

        nodules = (
            pd.merge(nodules, nodules_mapping,
                     left_on='AccessionNumber',
                     right_on='AccessionNumber')
            .rename(columns={'Index': 'seriesuid'})
        )
    return dataset, nodules


def save_histo(histo, path):
    with open(path, 'wb') as file:
        pickle.dump(histo, file)


def load_histo(path):
    with open(path, 'rb') as file:
        histo = pickle.load(file)
    return histo
