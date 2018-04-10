""" Contains functions that can be helpful for initial CT-dataset manipulations. """
import os
from binascii import hexlify
from collections import OrderedDict
import glob
import numpy as np
import pandas as pd
import dicom


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

    nodule_type = nodule_type.str.replace(r'каверна', 'c')
    nodule_type = nodule_type.str.replace(r'nan', 'c')

    nodule_type = nodule_type.str.replace(r'g|ч|x', 'п')
    nodule_type = nodule_type.str.replace(r'а|a|k|к|‘', 'с')
    nodule_type = nodule_type.str.replace(r'v', 'м')
    nodule_type = nodule_type.str.replace(r'^$', 'c')

    nodule_type = nodule_type.str.replace(r'(?:n|п)(?:c|с)?', 'semi_solid')
    nodule_type = nodule_type.str.replace(r'м', 'ground_glass')
    nodule_type = nodule_type.str.replace(r'(?:c|с)(\d|\s.+)?', 'solid')

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


def read_nodules(path, include_annotators=False, drop_no_cancer=False):
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
