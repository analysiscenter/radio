""" Contains losses used in tensorflow models. """

import tensorflow as tf
from ...dataset.dataset.models.tf.losses import dice

dice_loss = dice


def reg_l2_loss(labels, predictions, lambda_coords=0.75):
    """ L2 loss for prediction of cancer tumor's centers, sizes joined with binary classification task.

    Parameters
    ----------
    labels : tf.Tensor
        tensor containing true values for sizes of nodules, their centers
        and classes of crop(1 if cancerous 0 otherwise).
    predictions : tf.Tensor
        tensor containing predicted values for sizes of nodules, their centers
        and probability of cancer in given crop.

    Returns
    -------
    tf.Tensor
        l2 loss for regression of cancer tumor center's coordinates,
        sizes joined with binary classification task.

    Notes
    -----
    labels and predictions tensors must have [None, 7] shapes;
    labels[:, :3] and predictions[:, :3] correspond to normalized (from [0, 1] interval)
    zyx coordinates of cancer tumor, while labels[:, 3:6] and predictions[:, 3:6]
    correspond to sizes of cancer tumor along zyx axes(also normalized),
    finally, labels[:, 6] and predictions[:, 6] represent whether cancer tumor presents
    or not in the current crop.
    """
    clf_true, clf_pred = labels[:, 6], predictions[:, 6]
    centers_true, centers_pred = labels[:, :3], predictions[:, :3]
    sizes_true, sizes_pred = labels[:, 3:6], predictions[:, 3:6]

    centers_loss = 0.5 * tf.reduce_sum((centers_true - centers_pred) ** 2, axis=1)
    sizes_loss = 0.5 * tf.reduce_sum((tf.sqrt(sizes_true) - tf.sqrt(sizes_pred)) ** 2, axis=1)
    clf_loss = 0.5 * (clf_true - clf_pred) ** 2

    loss = clf_loss + lambda_coords * clf_true * (centers_loss + sizes_loss)
    return tf.reduce_mean(loss)


def iou_3d(labels, predictions, epsilon=10e-7):
    """ Compute intersection over union in 3D case for input tensors.

    Parameters
    ----------
    labels : tf.Tensor
        tensor containg true values for sizes of nodules and their centers.
    predictions : tf.Tensor
        tensor containing predicted values for sizes of nodules and their centers.
    epsilon : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    tf.Tensor
        tensor containing intersection over union computed on input tensors.
    """
    with tf.variable_scope('IOU'):
        tf_epsilon = tf.constant([epsilon], tf.float32)
        r_true = labels[:, :3]
        r_pred = predictions[:, :3]

        s_true = labels[:, 3:6]
        s_pred = predictions[:, 3:6]

        abs_r_diff = tf.abs(r_true - r_pred)
        abs_s_diff = tf.abs(s_true - s_pred)

        iou_tensor = tf.where(abs_r_diff < abs_s_diff, 2 * tf.minimum(s_true, s_pred),
                              tf.clip_by_value(s_true + s_pred - abs_r_diff, 0, 1))

        iou_tensor = (tf.reduce_prod(iou_tensor, axis=1)
                      / (tf.reduce_prod(s_true, axis=1)
                         + tf.reduce_prod(s_pred, axis=1) + tf_epsilon))
    return iou_tensor


def tversky_loss(labels, predictions, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.

    Parameters
    ----------
    labels : tf.Tensor
        tensor containing target mask.
    predictions : tf.Tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    tf.Tensor
        tensor containing tversky loss.
    """
    labels = tf.contrib.layers.flatten(labels)
    predictions = tf.contrib.layers.flatten(predictions)
    truepos = tf.reduce_sum(labels * predictions)
    fp_and_fn = (alpha * tf.reduce_sum(predictions * (1 - labels))
                 + beta * tf.reduce_sum((1 - predictions) * labels))

    return -(truepos + smooth) / (truepos + smooth + fp_and_fn)


def jaccard_coef_logloss(labels, predictions, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    labels : tf.Tensor
        tensor containing target mask.
    predictions : tf.Tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    tf.Tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    labels = tf.contrib.layers.flatten(labels)
    predictions = tf.contrib.layers.flatten(predictions)
    truepos = tf.reduce_sum(labels * predictions)
    falsepos = tf.reduce_sum(predictions) - truepos
    falseneg = tf.reduce_sum(labels) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -tf.log(jaccard + smooth)
