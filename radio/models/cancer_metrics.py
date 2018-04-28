"""cancer metrics"""
import numpy as np
from tqdm import tqdm_notebook as tqn
from scipy.ndimage.measurements import label


def binarization(arr, threshold=.5):
    """ Binarizate input data by theshold. Input data can be array of data of different sizes.

    Parameters
    ----------
    arr : list or np.array
        array with data to binarizate
    threshold : float
        The threshold of binarization.

    Returns
    -------
        binarization numpy array with the same size as arr
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    return np.where(arr > threshold, 1, 0).astype(np.uint8)

def get_connected_ixs(array):
    """ Find connected components in the array.

    Parameters
    ----------
    array : list or np.array
        input array

    Returns
    -------
        An array of the same size as the input, in which the index of the given
        component stands in the place of the set of connected components
    """
    connected_array = label(array)
    if connected_array[1] == 0:
        return None
    ixs = [np.concatenate(np.where(connected_array[0] == i)).reshape(4, -1) for i in range(1, connected_array[1]+1)]
    return ixs

def sensitivity_number(target, prediction, threshold=.5, iot=.5, **kwargs):
    """ True positive rate. Сalculates the percentage of correctly predicted values by number of nodules.

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct

    Returns
    -------
        Percentage of correctly predicted values
    """
    target, prediction = binarization([target, prediction], threshold)
    connected_target = get_connected_ixs(target)
    connected_prediction = get_connected_ixs(prediction)
    if connected_target is None:
        if len(connected_prediction) > 0:
            return 0
        return 1
    intersection = prediction * target
    right = 0
    for coord_nodule in connected_target:
        mass = [slice(np.min(coord_nodule[i]), np.max(coord_nodule[i]))
                if np.min(coord_nodule[i]) != np.max(coord_nodule[i])
                else slice(np.min(coord_nodule[i]), np.max(coord_nodule[i])+1)
                for i in range(coord_nodule.shape[0])]

        nodule_perc = intersection[mass]
        if np.sum(nodule_perc)/len(nodule_perc) > iot:
            right += 1
    total = right / len(connected_target)
    return total

def sensitivity_volume(target, prediction, threshold=.5, iot=.5, **kwargs):
    """ True positive rate. Сalculates the percentage of correctly predicted values by volume of nodules.

    Parameters
    ----------
    target : Pandas DataFrame
        DataFrame with target parameters of noudles.
    prediction : Pandas DataFrame
        DataFrame with predicted parameters of noudles.
    threshold : float
        The threshold of binarization
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct

    Returns
    -------
        Percentage of correctly predicted values
    """
    predicted_nodules = 0
    for ix in target.iterrows():
        true_diam, true_x, true_y, true_z = ix[1][target.columns[:4]]
        try:
            pred_diam, pred_x, pred_y, pred_z = prediction.loc[ix[1]['overlap_index']][:4]
        except TypeError:
            pass
        dist = (np.abs(pred_x - true_x)**2 + np.abs(pred_y - true_y) + np.abs(pred_z - true_z)**2)**.5
        r = pred_diam / 2
        R = true_diam / 2
        V = (np.pi * (R + r - dist)**2 * (dist**2 + 2*dist*r - 3*r*r + 2*dist*R + 6*r*R - 3*R*2)) / (12*dist)
        V_true = 4/3 * np.pi * R**3
        if V/V_true > iot:
            predicted_nodules += 1
    return predicted_nodules/len(target)

def false_positive_number(target, prediction, threshold=.5, iot=.5, **kwargs):
    """ Calculate the number of falsely predicted values.

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct

    Returns
    -------
        The number of falsely predicted values
    """
    target, prediction = binarization([target, prediction], threshold)
    connected_target = get_connected_ixs(target)
    connected_prediction = get_connected_ixs(prediction)

    if connected_target is None:
        if len(connected_prediction) > 0:
            return 0
        return 1

    intersection = prediction * target
    wrong_nodules = len(connected_prediction)

    for coord_nodule in connected_target:
        mass = [slice(np.min(coord_nodule[i]), np.max(coord_nodule[i]))
                if np.min(coord_nodule[i]) != np.max(coord_nodule[i])
                else slice(np.min(coord_nodule[i]), np.max(coord_nodule[i])+1)
                for i in range(coord_nodule.shape[0])]

        nodule_perc = intersection[mass]
        if np.sum(nodule_perc)/len(nodule_perc) > iot:
            wrong_nodules -= 1
    return wrong_nodules

def specificity(target, prediction, threshold=.5, **kwargs):
    """ True negative rate. Calculate according to the formula:
            S = sum((1 - target) * (1 - prediction)) / sum(1 - target)

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization

    Returns
    -------
        The value of true negative rate
    """
    target, prediction = binarization([target, prediction], threshold)
    total = np.sum((1 - target) * (1 - prediction)) / np.sum(1 - target)
    return total

def false_discovery_rate(target, prediction, threshold=.5, **kwargs):
    """ False discovery rate. Calculate according to the formula:
            rate = sum((1 - target) * prediction) / sum(prediction)

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization

    Returns
    -------
        The value of false discovery rate
    """
    target, prediction = binarization([target, prediction], threshold)
    rate = np.sum((1 - target) * prediction) / np.sum(prediction)
    return rate

def positive_likelihood_ratio(target, prediction, threshold=.5, iot=.5, sens_type='n', **kwargs):
    """ Positive likelihood ratio. Calculate according to the formula:
            rato = (sensitivity) / (1 - specificity)

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct
    sens_type : char
        if 'n' sesitivity_number will be used. If 'v' sesitivity_values will be used instead.
    Returns
    -------
        The value of positive likelihood ratio.
    """
    sensitivity = sensitivity_number if sens_type == 'n' else sensitivity_volume
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    ratio = (sens) / ((1 - spec) + 1e-9)
    return ratio

def negative_likelihood_ratio(target, prediction, threshold=.5, iot=.5, sens_type='n', **kwargs):
    """ Negative likelihood ratio. Calculate according to the formula:
            rato = (specificity) / (1 - sensitivity)

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct
    sens_type : char
        if 'n' sesitivity_number will be used. If 'v' sesitivity_values will be used instead.
    Returns
    -------
        The value of negative likelihood ratio
    """
    sensitivity = sensitivity_number if sens_type == 'n' else sensitivity_volume
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    ratio = (spec) / ((1 - sens) + 1e-9)
    return ratio

def froc(target, prediction, threshold=.5, iot=.5, n_points=50, sens_type='n', **kwargs):
    """ Draws a graph of the dependence of specificity and the number of false positive values
    from the threshold.

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    threshold : float
        The threshold of binarization
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct
    n_points : int
        The number of points on the curve.
    sens_type : char
        if 'n' sesitivity_number will be used. If 'v' sesitivity_values will be used instead.

    Returns
    -------
    Array with len of 2: [sensitivity, number of false positive values]
    """
    target, prediction = binarization([target, prediction], threshold)
    sens = []
    false_pos = []
    sensitivity = sensitivity_number if sens_type == 'n' else sensitivity_volume
    for thr in tqn(np.linspace(0, 1, n_points)):
        sens.append(sensitivity(target, prediction, threshold=thr, iot=iot))
        false_pos.append(false_positive_number(target, prediction, threshold=thr, iot=iot))
    return [sens, false_pos]

METRICS = {
    'sensitivity_volume': sensitivity_volume,
    'sensitivity_number': sensitivity_number,
    'false_positive_number': false_positive_number,
    'specificity': specificity,
    'false_discovery_rate': false_discovery_rate,
    'positive_likelihood_ratio': positive_likelihood_ratio,
    'negative_likelihood_ratio': negative_likelihood_ratio,
    'froc': froc
}

def calcualte_metrics(target, prediction, metrics, **kwargs):
    """Calculated metrics by given array with names of metrics

    Parameters
    ----------
    target : list or np.array
        An array containing the target values
    prediction : list or np.array
        An array containing the predicted values
    metrics : list or np.array
        An array with names of metrics to calculate

    Returns
    -------
    dict
        Dict with metric names as keys and values as a result of calculating metrics.
    """
    calculated_metrics = {}
    for metric in metrics:
        calculated_metrics[metric] = METRICS[metric](target, prediction, **kwargs)
    return calculated_metrics
