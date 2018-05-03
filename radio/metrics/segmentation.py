""" Contains metrics """
import numpy as np
from scipy.ndimage import measurements


def binarize(mask, threshold=.5):
    """ Create a binary mask from probabilities with a given threshold.

    Parameters
    ----------
    mask : np.array
        input mask with probabilities
    threshold : float
        where probability is above the threshold, the output mask will have 1, otherwise 0.

    Returns
    -------
    np.array
        binary mask of the same shape as the input mask
    """
    mask = np.asarray(mask)
    return np.where(mask >= threshold, 1, 0).astype(np.uint8)


def get_nodules(mask):
    """ Find nodules as connected components in the input mask.

    Parameters
    ----------
    mask : np.array
        Binary mask

    Returns
    -------
    np.array or None
        An array of coords of the nodules found
    """
    connected_array, num_components = measurements.label(mask, output=None)
    if num_components == 0:
        return None
    nodules = []
    for i in range(1, num_components + 1):
        coords = [slice(np.min(c), np.max(c) + 1) for c in np.where(connected_array == i)]
        nodules.append(coords)
    return np.array(nodules)


def sensitivity(target, prediction, threshold=.5, **kwargs):
    """ True positive rate (by volume).

    Сalculates the percentage of correctly predicted masked pixels.

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold

    Returns
    -------
    float or None
        The percentage of correctly predicted values.
        None if there is nothing to predict (target contains zeros).
    """
    target, prediction = binarize([target, prediction], threshold)
    total_target = np.sum(target)
    if total_target > 0:
        total = np.sum(target * prediction) / total_target
    else:
        total = None
    return total


def sensitivity_nodules(target, prediction, threshold=.5, iot=.5, **kwargs):
    """ True positive rate (by nodules).

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold
    iot : float
        The percentage of intersection between the predicted and the target nodules,
        at which the prediction is counted as correct

    Returns
    -------
    float or None
        The percentage of correctly predicted nodules
        None if there is nothing to predict (target contains zeros).
    """
    target = binarize(target, threshold)

    if np.sum(target) == 0:
        total = None
    else:
        target_nodules = get_nodules(target)
        intersection = prediction * target

        right = 0
        for coord in target_nodules:
            predicted_nodule = intersection[coord]
            if np.sum(predicted_nodule) / predicted_nodule.size >= iot:
                right += 1
        total = right / len(target_nodules)

    return total


def false_positive_nodules(target, prediction, threshold=.5, iot=.5, **kwargs):
    """ Calculate the number of falsely predicted nodules.

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold
    iot : float
        The percentage of intersection between the predicted and the target nodules,
        at which the prediction is counted as correct

    Returns
    -------
    int
        The number of falsely predicted nodules
    """
    prediction = binarize(prediction, threshold)

    if np.sum(prediction) == 0:
        total = 0
    else:
        predicted_nodules = get_nodules(prediction)
        target = binarize(target, threshold)

        total = 0
        for coord in predicted_nodules:
            nodule_true_mask = target[coord]
            if np.sum(nodule_true_mask) / nodule_true_mask.size < iot:
                total += 1

    return total


def specificity(target, prediction, threshold=.5, **kwargs):
    """ True negative rate (by volume)

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold

    Returns
    -------
    float or None
        True negative rate
        None if there is nothing to predict (target contains only ones).
    """
    target, prediction = binarize([target, prediction], threshold)
    total_target = np.sum(1 - target)
    if total_target > 0:
        total = np.sum((1 - target) * (1 - prediction)) / total_target
    else:
        total = None
    return total


def false_discovery_rate(target, prediction, threshold=.5, **kwargs):
    """ False discovery rate (by volume)

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold

    Returns
    -------
    float
        False discovery rate
    """
    target, prediction = binarize([target, prediction], threshold)
    total_prediction = np.sum(prediction)
    if total_prediction > 0:
        rate = np.sum((1 - target) * prediction) / total_prediction
    else:
        rate = 0.
    return rate


def positive_likelihood_ratio(target, prediction, threshold=.5, **kwargs):
    """ Positive likelihood ratio.

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct

    Returns
    -------
    float or None
        Positive likelihood ratio.
    """
    e = 1e-15
    sens = sensitivity(target, prediction, threshold)
    spec = specificity(target, prediction, threshold)
    if sens is not None and spec is not None:
        ratio = sens / (1 - spec + e)
    else:
        ratio = None
    return ratio


def negative_likelihood_ratio(target, prediction, threshold=.5, **kwargs):
    """ Negative likelihood ratio.

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    threshold : float
        Binarization threshold
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct

    Returns
    -------
    float or None
        Negative likelihood ratio
    """
    e = 1e-15
    sens = sensitivity(target, prediction, threshold, iot)
    spec = specificity(target, prediction, threshold)
    if sens is not None and spec is not None:
        ratio = (1 - sens) / (spec + e)
    else:
        ratio = None
    return ratio


def froc(target, prediction, n_points=50, threshold=.5, iot=.5, **kwargs):
    """ Calculate Free-reponse ROC: specificity vs the number of false positives per image.

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    n_points : int
        The number of points on the curve
    threshold : float
        Binarization threshold
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct

    Returns
    -------
    tuple of 2 np.array
        sensitivity, number of false positives
    """
    target, prediction = binarize([target, prediction], threshold)
    sens = []
    false_pos = []
    for threshold in np.linspace(0, 1, n_points):
        sens.append(sensitivity(target, prediction, threshold=threshold, iot=iot))
        false_pos.append(false_positive_nodules(target, prediction, threshold=threshold, iot=iot))
    return sens, false_pos
