""" Contains ratios metrics """

def positive_likelihood_ratio(sensitivity, specificity, **kwargs):
    """ Positive likelihood ratio.

    Parameters
    ----------
    sensitivity : float
        The value of any sensitivity metrics (by volume, by nodules, etc.)
    specificity : float
        The value of any specificity metrics (by volume, etc.)

    Returns
    -------
    float or None
        Positive likelihood ratio.
    """
    e = 1e-15
    if sensitivity is not None and specificity is not None:
        ratio = sensitivity / (1 - specificity + e)
    else:
        ratio = None
    return ratio


def negative_likelihood_ratio(sensitivity, specificity, **kwargs):
    """ Negative likelihood ratio.

    Parameters
    ----------
    sensitivity : float
        The value of any sensitivity metrics (by volume, by nodules, etc.)
    specificity : float
        The value of any specificity metrics (by volume, etc.)

    Returns
    -------
    float or None
        Negative likelihood ratio
    """
    e = 1e-15
    if sensitivity is not None and specificity is not None:
        ratio = (1 - sensitivity) / (specificity + e)
    else:
        ratio = None
    return ratio

