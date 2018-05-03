from .segmentation import sensitivity, sensitivity_nodules, false_positive_nodules, specificity,
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
        The binarize threshold
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


def calculate_metrics(targets, predictions, metrics=None, threshold=.5, iot=.5):
    """ Evaluate all metrics """
    for i in len(targets):
        metrics_one = _calculate_metrics(targets[i], predictions[i], metrics=metrics,
                                         threshold=threshold, iot=iot)
