""" Contain metrics and metric calculation routines """
import numpy as np

from .segmentation import sensitivity, sensitivity_nodules, false_positive_nodules, specificity, \
                          false_discovery_rate, positive_likelihood_ratio, negative_likelihood_ratio, froc


METRICS = {
    'sensitivity': sensitivity,
    'sensitivity_nodules': sensitivity_nodules,
    'false_positive_nodules': false_positive_nodules,
    'specificity': specificity,
    'false_discovery_rate': false_discovery_rate,
    'positive_likelihood_ratio': positive_likelihood_ratio,
    'negative_likelihood_ratio': negative_likelihood_ratio,
    'froc': froc
}


def _calculate_metrics(target, prediction, metrics, **kwargs):
    """ Calculated metrics for one item

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
    metrics : list of str
        Names of metrics to calculate

    Returns
    -------
    dict
        Dict with metric names as keys and values as a result of calculating metrics.
    """
    calculated_metrics = {}
    for metric in metrics:
        calculated_metrics[metric] = METRICS[metric](target, prediction, **kwargs)
    return calculated_metrics


def aggregate_metrics(metrics, method='mean'):
    """ Calculate accumulated metrics from item metrics.

    Parameters
    ----------
    metrics : list of dict
        A collection of metrics for individual items
    method : 'mean' or callable
        A function to aggregate individual metrics

    Returns
    -------
    dict
        Metric names as keys and calculated metrics as values.
    """
    if method == 'mean':
        method = np.mean

        all_metrics = {}
        for name in metrics:
            all_metrics[name] = method(metrics[name])
    elif callable(method):
        all_metrics = method(metrics)
    else:
        raise ValueError("An aggregation method should be 'mean' or a callable.", method)

    return all_metrics


def calculate_metrics(targets, predictions, threshold=.5, iot=.5, metrics=None, agg=None):
    """ Evaluate metrics for all targets / predictions

    Parameters
    ----------
    target : np.array
        An array of target masks
    prediction : np.array
        An array of predicted masks
    threshold : float
        Binarization threshold
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct
    metrics : list of str
        Names of metrics to calculate
    agg : 'mean' or callable
        An aggregation method.
        If None, returns a list of metrics dicts - one for each predicted item.

    Returns
    -------
    dict or list of dict
        Metrics names as keys and calculated metrics as values.
    """
    if metrics is None:
        metrics = list(METRICS.keys())

    _metrics = []
    for i in range(len(targets)):
        metrics_one = _calculate_metrics(targets[i], predictions[i], metrics=metrics,
                                         threshold=threshold, iot=iot)
        _metrics.append(metrics_one)

    if agg is None:
        all_metrics = _metrics
    else:
        all_metrics = aggregate_metrics(_metrics, method=agg)

    return all_metrics


