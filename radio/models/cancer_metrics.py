"""cancer metrics"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqn
from scipy.ndimage.measurements import label


def binarization(arr, threshold=.5, upper=1, lower=0):
    """binarizate arr by theshold"""
    if isinstance(arr, list):
        arr = np.array(arr)
    return np.where(arr > threshold, upper, lower).astype(np.uint8)

def get_connected_ixs(array):
    """find connected components in the array"""
    connected_array = label(array)
    if connected_array[1] == 0:
        return None
    ixs = [np.concatenate(np.where(connected_array[0] == i)).reshape(4, -1) for i in range(1, connected_array[1]+1)]
    return ixs

def sensitivity(target, prediction, threshold=.5, iot=.5, **kwargs):
    """True positive rate"""
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

def false_positive_per_scan(target, prediction, threshold=.5, iot=.5, **kwargs):
    """number of falsely predicted nodules on scan"""
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
    """True negative rate"""
    target, prediction = binarization([target, prediction], threshold)
    total = np.sum((1 - target) * (1 - prediction)) / np.sum(1 - target)
    return total

def false_discovery_rate(target, prediction, threshold=.5, **kwargs):
    """false_discovery_rate """
    target, prediction = binarization([target, prediction], threshold)
    total = np.sum((1 - target) * prediction)/np.sum(prediction)
    return total

def posivite_likelihood_rato(target, prediction, threshold=.5, iot=.5, **kwargs):
    """posivite_likelihood_rato """
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    rato = (sens) / (1 - spec)
    return rato

def negative_likelihood_ratio(target, prediction, threshold=.5, iot=.5, **kwargs):
    """negative_likelihood_ratio """
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    rato = (spec) / (1 - sens)
    return rato

def froc(target, prediction, threshold=.5, iot=.5, **kwargs):
    """froc """
    target, prediction = binarization([target, prediction], threshold)
    sens = []
    false_pos = []
    for thr in tqn(np.linspace(0, 1)):
        sens.append(sensitivity(target, prediction, threshold=thr, iot=iot))
        false_pos.append(false_positive_per_scan(target, prediction, threshold=thr, iot=iot))
    plt.plot(false_pos, sens)
    plt.show()

METRICS = {
    'sensitivity': sensitivity,
    'false_positive_per_scan': false_positive_per_scan,
    'specificity': specificity,
    'false_discovery_rate': false_discovery_rate,
    'posivite_likelihood_rato': posivite_likelihood_rato,
    'negative_likelihood_ratio': negative_likelihood_ratio,
    'froc': froc
}

def calcualte_metrics(target, prediction, metrics, **kwargs):
    """calculated metrics by given metrics"""
    calculated_metrics = {}
    for metric in metrics:
        calculated_metrics[metric] = METRICS[metric](target, prediction, **kwargs)
    return calculated_metrics
