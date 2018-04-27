"""cancer metrics"""
import numpy as np
import matplotlib.pyplot as plt
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

def sensitivity(target, prediction, threshold=.5, iot=.5, **kwargs):
    """ True positive rate. Сalculates the percentage of correctly predicted values.

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

def positive_likelihood_ratio(target, prediction, threshold=.5, iot=.5, **kwargs):
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

    Returns
    -------
        The value of positive likelihood ratio.
    """
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    ratio = (sens) / ((1 - spec) + 1e-9)
    return ratio

def negative_likelihood_ratio(target, prediction, threshold=.5, iot=.5, **kwargs):
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

    Returns
    -------
        The value of negative likelihood ratio
    """
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    ratio = (spec) / ((1 - sens) + 1e-9)
    return ratio

def froc(target, prediction, threshold=.5, iot=.5, **kwargs):
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
    """
    target, prediction = binarization([target, prediction], threshold)
    sens = []
    false_pos = []
    for thr in tqn(np.linspace(0, 1)):
        sens.append(sensitivity(target, prediction, threshold=thr, iot=iot))
        false_pos.append(false_positive_number(target, prediction, threshold=thr, iot=iot))
    plt.plot(false_pos, sens)
    plt.show()

METRICS = {
    'sensitivity': sensitivity,
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
