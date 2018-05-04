""" Contains ROC metrics """
import numpy as np
from .core import binarize


def froc(target, prediction, sensitivity=None, false_positive=None, n_points=50, threshold=.5, iot=.5, **kwargs):
    """ Calculate Free-reponse ROC: specificity vs the number of false positives per image.

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    sensitivity : callable
        The function to calculate sensitivity metrics (by volume, by nodules, etc.)
    false_positive : callable
        The function to calculate the number of false positives (by volume, by nodules, etc.)
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
        false_pos.append(false_positive(target, prediction, threshold=threshold, iot=iot))
    return sens, false_pos
