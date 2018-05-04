""" Contain metrics and metric calculation routines """
import numpy as np

from .core import binarize, get_nodules
from .mask import MetricsByNodules, MetricsByVolume
from .roc import froc
from .ratio import negative_likelihood_ratio, positive_likelihood_ratio


def _calculate_metrics(target, prediction, *args, metrics=None, bin=False, **kwargs):
    """ Calculated metrics for one item

    Parameters
    ----------
    target : np.array
        Target mask
    prediction : np.array
        Predicted mask
    metrics : list of str
        Names of metrics to calculate
    bin : bool
        Whether to binarize arrays before metrics calculation
    threshold : float
        Binarization threshold (default=0.5)
    iot : float
        Percentage of intersection between the prediction and the target,
        at which the prediction is interpreted as correct (default=1e-6)

    Returns
    -------
    dict
        Dict with metric names as keys and values as a result of calculating metrics.
    """
    if metrics is None:
        raise ValueError('Explicitly specify a list of metrics to calculate', metrics)
    if 'threshold' not in kwargs:
        kwargs['threshold'] = .5
    if 'iot' not in kwargs:
        kwargs['iot'] = 1e-6
    target, prediction = binarize([target, prediction], kwargs['threshold'])

    metrics_fn = {}
    for name in metrics:
        _m = name.replace(" ", "")
        if '/nodules' in _m:
            src = MetricsByNodules
        elif '/volume' in _m:
            src = MetricsByVolume
        else:
            raise ValueError("Unknown metrics", m)

        name = _m.split('/')[0]
        if hasattr(src, name):
            metrics_fn[m] = getattr(src, name)
        else:
            raise ValueError("Metrics has not been found", m)

    if bin:
        target, prediction = binarize([target, prediction], kwargs['threshold'])

    calculated_metrics = {}
    for name, method in metrics_fn.items():
        calculated_metrics[name] = method(target, prediction, **kwargs)
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
        for item in metrics:
            for name, value in item.items():
                if name not in all_metrics:
                    all_metrics[name] = list()
                if value is not None:
                    all_metrics[name].append(value)
        for name, value in all_metrics.items():
            all_metrics[name] = method(value)
    elif callable(method):
        all_metrics = method(metrics)
    else:
        raise ValueError("An aggregation method should be 'mean' or a callable.", method)

    return all_metrics


def calculate_metrics(targets, predictions, metrics=None, agg=None, **kwargs):
    """ Evaluate metrics for all targets / predictions

    Parameters
    ----------
    target : np.array
        An array of target masks
    prediction : np.array
        An array of predicted masks
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
        raise ValueError('Explicitly specify a list of metrics to calculate', metrics)

    _metrics = []
    for i, _ in enumerate(targets):
        metrics_one = _calculate_metrics(targets[i], predictions[i], metrics=metrics, **kwargs)
        _metrics.append(metrics_one)

    if agg is None:
        return _metrics

    all_metrics = aggregate_metrics(_metrics, method=agg)

    return all_metrics
