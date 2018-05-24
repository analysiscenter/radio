""" Contains functions that can be helpful for initial CT-dataset manipulations. """
import os
from binascii import hexlify
from collections import OrderedDict
import glob
import numpy as np
import pandas as pd
import dicom
import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import pickle

def generate_index(size=20):
    """ Generate random string index of givne size.

    Parameters
    ----------
    size : int
        length of index string.

    Returns
    -------
    str
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

def get_dicom_origin(path):
    """ Get origin for dicom from path """
    def read_dicom(x):
        return dicom.read_file(os.path.join(path, x))
    pool = ThreadPool()
    results = pool.map(read_dicom, os.listdir(path))
    pool.close()
    pool.join()
    list_of_dicoms = list(results)

    list_of_dicoms.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)

    dicom_slice = list_of_dicoms[0]
    origin = np.asarray([float(dicom_slice.ImagePositionPatient[2]),
                         float(dicom_slice.ImagePositionPatient[0]),
                         float(dicom_slice.ImagePositionPatient[1])], dtype=np.float)
    return origin


def get_dicom_info(paths, index_col=None, progress=False):
    """ Accumulates scans meta information from dicom dataset in pandas DataFrame.

    Parameters
    ----------
    paths : list, tuple or ndarray of strings
        paths to directories with dicom files.
    index_col : str or None
        name of column that will be used as index of output DataFrame.
    progress : bool
        show progress bar or not

    Returns
    -------
    pandas.DataFrame
        pandas DataFrame with scans' meta information.
    """
    meta_info = []
    progress = tqdm.tqdm if progress else lambda x: x
    for path in progress(paths):
        first_slice = dicom.read_file(os.path.join(path, os.listdir(path)[0]))
        origins = get_dicom_origin(path)
        info_dict = {
            'SpacingZ': float(first_slice.SliceThickness),
            'SpacingY': float(first_slice.PixelSpacing[0]),
            'SpacingX': float(first_slice.PixelSpacing[1]),
            'StudyID': str(first_slice.StudyID),
            'seriesid': str(first_slice.AccessionNumber),
            'PatientID': str(first_slice.PatientID),
            'Rows': int(first_slice.Rows),
            'Columns': int(first_slice.Columns),
            'NumSlices': len(os.listdir(path)),
            'ScanID': os.path.basename(path),
            'Index': str(first_slice.AccessionNumber) + '_' + os.path.basename(path),
            'ScanPath': path,
            'OriginZ': origins[0],
            'OriginY': origins[1],
            'OriginX': origins[2]
        }
        meta_info.append(info_dict)
    return pd.DataFrame(meta_info) if index_col is None else pd.DataFrame(meta_info).set_index(index_col)

def get_blosc_info(paths, index_col=None, progress=False):
    """ Accumulates scans meta information from dicom dataset in pandas DataFrame.

    Parameters
    ----------
    paths : list, tuple or ndarray of strings
        paths to directories with dicom files.
    index_col : str or None
        name of column that will be used as index of output DataFrame.
    progress : bool
        show progress bar or not

    Returns
    -------
    DataFrame
        pandas DataFrame with scans' meta information.
    """
    meta_info = []
    progress = tqdm.tqdm if progress else lambda x: x
    for path in progress(paths):
        results = []
        for component in ['spacing', 'origin']:
            with open(os.path.join(path, component, 'data.pkl'), 'rb') as f:
                results.append(pickle.load(f))
        spacing, origin = results
        info_dict = {
            'SpacingZ': spacing[0][0],
            'SpacingY': spacing[0][1],
            'SpacingX': spacing[0][2],
            'OriginZ': origin[0][0],
            'OriginY': origin[0][1],
            'OriginX': origin[0][2],
            'seriesid': path.split('/')[-1],
            'ScanPath': path,
            'ScanID': os.path.basename(path)
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
    pandas.DataFrame
    """
    output_indices = (
        info_df
        .groupby('seriesid')
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
    pandas.DataFrame
        annotation.
    """
    base_columns = ['seriesid', 'StudyID', 'DoctorID', 'Comment', 'NumNodules']
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
    for group in annotation_df.groupby(['seriesid', 'DoctorID']):
        accession_number = group[0][0]
        doctor_id = group[0][1]

        nodules = group[1].iloc[:, 5:].values.reshape(-1, 5)
        for i in range(nodules.shape[0]):
            nodule_id = generate_index()
            nodule_dict = {
                'seriesid': accession_number,
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


def read_annotators_info(path, annotator_prefix=None, binary=True):
    """ Read information about annotators from annotation.

    This method reads information about annotators and scans into pandas DataFrame
    that contains accession numbers as indices and columns names
    corresponding to ids of annotators with prefix added (if provided). Each cell
    of the output table is filled with '1' if corresponding annotator
    was annotating scan with given accession number and '0' otherwise.

    Parameters
    ----------
    annotation : pandas DataFrame

    annotator_prefix : str or None
        prefix of annotators indices in the output DataFrame.

    Returns
    -------
    pandas.DataFrame
        table with of shape (num_scans, num_annotators) filled with '1'
        if annotator annotated given scan or '0' otherwise.
    """
    if binary:
        annotators_info = (
            parse_annotation(path)
            .assign(DoctorID=lambda df: df.DoctorID.str.replace("'", ""))
            .query("seriesid != ''")
            .drop_duplicates()
            .pivot('seriesid', 'DoctorID', 'DoctorID')
            .notna()
            .astype('int')
        )
        if annotator_prefix:
            annotators_indices_mapping = {index: (annotator_prefix + index)
                                          for index in annotators_info.columns}

            annotators_info = annotators_info.pipe(lambda df: df.rename(columns=annotators_indices_mapping))
    else:
        annotators_info = (annotation
                           .assign(DoctorID=lambda df: df.DoctorID.str.replace("'", ""))
                           .query("seriesid != ''"))[['seriesid', 'DoctorID']]
    return annotators_info
        


def read_nodules(path, include_annotators=False):
    """ Read annotation from file and transform it to DataFrame with nodules.

    Output DataFrame contains following columns:
    'NoduleID' unique id of nodule.
    'seriesid' - accession number of CT scan that corresponding to nodule.
    'DoctorID' - id of annotator.
    'coordZ', 'coordY', 'coordX' - coordinates of nodule's center.
    'diameter_mm' - diameter of nodule.
    'NoduleType' - type of nodule. Can be one of ['solid', 'semi_solid', 'ground_glass'].

    Unique id of nodule generated each time function called and is used as index
    of the output DataFrame.

    Parameters
    ----------
    path : str
        path to file with annotation
    include_annotators : bool
        include or not rows with nan for doctors who annotated image but didn't find
        nodules

    Returns
    -------
    pandas.DataFrame
        dataframe that contains information about nodules location, type and etc.
    """
    annotation = parse_annotation(path)
    nodules = annotation_to_nodules(annotation)
    if include_annotators:
        annotators_info = read_annotators_info(annotation, binary=False)
        nodules = nodules.merge(annotators_info, left_on=['seriesid', 'DoctorID'], 
                                right_on=['seriesid', 'DoctorID'], how='outer')
    return nodules

def read_dataset_info(path=None, paths=None, index_col=None, filter_by_min_spacing=False, fmt='dicom'):
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
    filter_by_min_spacing : bool
        for each accession number choose study with minimal spacing 
    fmt : str
        'dicom' or 'blosc'

    Returns
    -------
    pandas.DataFrame
        dataframe containing info about dicom dataset.
    """
    if (path is None and paths is None) or (path is not None and paths is not None):
        raise ValueError("Only one of 'path' or 'paths' arguments must be provided")

    if fmt == 'dicom':   
        dataset_info = get_dicom_info(glob.glob(path) if path is not None else paths)
        if filter_by_min_spacing:
            output_indices = (
                dataset_info
                .groupby('seriesid')
                .agg({'SpacingZ': 'idxmin'})
            )
            index_df = dataset_info.loc[output_indices.loc[:, 'SpacingZ'], :]
        else:
            index_df = dataset_info
    elif fmt == 'blosc':
        index_df = get_blosc_info(glob.glob(path) if path is not None else paths)
    else:
        raise ValueError('fmt must be dicom or blosc but {} were given'.format(fmt))
    return index_df if index_col is None else index_df.set_index(index_col)

def transform_annotation(annotation_path, images_path, fmt='dicom', include_annotators=True, drop=True):
    """ Transform annotation file to LUNA format with coordinates and diamters in mm.

    Parameters
    ----------
    annotation_path : str
        mask for txt files with annotation
    images_path : str
        mask for folders with dicom files
    fmt : str
        'dicom' or 'blosc'
    include_annotators : bool
        if doctor annotates image but don't find nodules, row with his ID, seriesid
        (seriesid) and other nans will be added
    drop : bool
        if True and file `annotation_path` has annotation for images which don't contain
        in folder `images_path`, it will be dropped
    
    Returns
    -------
    pandas.DataFrame
        dataframe with annotation with columns `[seriesid, DoctorID, NoduleID, coordX, coorY, coordZ, diam]`.
        Coordinates and diameter in mm.
    """
    annotations = glob.glob(annotation_path)
    nodules = pd.concat([read_nodules(annotation, include_annotators).reset_index(drop=True) for annotation in annotations]).reset_index(drop=True)
    paths = glob.glob(images_path)

    dataset_info = read_dataset_info(images_path, filter_by_min_spacing=True, fmt=fmt)
    spacing_info = dataset_info[['seriesid', 'SpacingX', 'SpacingY', 'SpacingZ',
                                 'OriginX', 'OriginY', 'OriginZ']].set_index('seriesid')
    nodules = nodules.join(spacing_info, on='seriesid', how='inner')

    for name in ['X', 'Y', 'Z']:
        nodules['coord'+name] = nodules['coord'+name] * nodules['Spacing'+name] + nodules['Origin'+name]
        del nodules['Spacing'+name]
        del nodules['Origin'+name]

    return nodules.reset_index(drop=True)