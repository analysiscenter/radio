""" Contains losses used in keras models. """
from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    """ Dice coefficient function implemente via keras backend.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target masks.
    y_pred : keras tensor
        tensor containing predicted masks.

    Returns
    -------
    keras tensor
        tensor containing dice coefficient value.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return answer


def dice_coef_loss(y_true, y_pred):
    """ Loss function base on dice coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.

    Returns
    -------
    keras tensor
        tensor containing tiversky loss.
    """
    answer = -dice_coef(y_true, y_pred)
    return answer


def tiversky_coef(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tiversky coefficient.

    Parameters
    ----------
    y_true : keras tensor
        containing target masks.
    y_pred : keras tensor
        containing predicted masks.

    Returns
    -------
    keras tensor
        tensor containing tiverky coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return answer

def tiversky_loss(y_true, y_pred):
    """ Tiversky loss function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.

    Returns
    -------
    keras tensor
        tensor containing tiversky loss.
    """
    return -tiversky_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred, smooth=1e-10):
    """ Jaccard coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing actual pixel-by-pixel values for all classes.
    y_pred : keras tensor
        tensor containing predicted pixel-by-pixel values for all classes.

    Returns
    -------
    keras tensor
        tensor containing jaccard score across all classes.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    return (truepos + smooth) / (smooth + truepos + falseneg + falsepos)


def jaccard_coef_logloss(y_true, y_pred):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.

    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient;
    """
    return -K.log(jaccard_coef(y_true, y_pred))
