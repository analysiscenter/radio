""" Contains losses used in keras models. """
from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    """ Dice coefficient required by keras model as a part of loss function.

    Args:
    - y_true: keras tensor with targets;
    - y_pred: keras tensor with predictions;

    Returns:
    - keras tensor with dice coefficient value;
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return answer


def dice_coef_loss(y_true, y_pred):
    """ Dice loss function.

    Args:
    - y_true: keras tensor containing target values;
    - y_pred: keras tensor containing predicted values;

    Returns:
    - keras tensor containing dice loss;
    """
    answer = -dice_coef(y_true, y_pred)
    return answer


def tiversky_coef(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tiversky coefficient.

    Args:
    - y_true: keras tensor containing target values;
    - y_pred: keras tensor containing predicted values;

    Returns:
    - keras tensor containing tiversky coefficient;
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return answer

def tiversky_loss(y_true, y_pred):
    """ Tiversky loss function.

    Args:
    - y_true: keras tensor containing target mask;
    - y_pred: keras tensor containing predicted mask;

    Returns:
    - keras tensor containing tiversky loss;
    """
    return -tiversky_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred, smooth=1e-10):
    """ Jaccard coefficient.

    Args:
    - y_true: actual pixel-by-pixel values for all classes;
    - y_pred: predicted pixel-by-pixel values for all classes;

    Returns:
    - jaccard score across all classes;
    """

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    return (truepos + smooth) / (smooth + truepos + falseneg + falsepos)


def jaccard_coef_logloss(y_true, y_pred):
    """ Keras loss function based on jaccard coefficient.

    Args:
    - y_true: keras tensor containing target mask;
    - y_pred: keras tensor containing predicted mask;

    Returns:
    - keras tensor with jaccard loss;
    """
    return -K.log(jaccard_coef(y_true, y_pred))
