""" Contains losses used in tensorflow models. """

import tensorflow as tf


def reg_l2_loss(y_true, y_pred, lambda_coords=0.75):
    """ L2 loss for prediction of cancer tumor's centers, sizes joined with binary classification task.

    Parameters
    ----------
    y_true : tf.Tensor
        tensor containing true values for sizes of nodules, their centers
        and classes of crop(1 if cancerous 0 otherwise).
    y_pred : tf.Tensor
        tensor containing predicted values for sizes of nodules, their centers
        and probability of cancer in given crop.

    Returns
    -------
    tf.Tensor
        l2 loss for regression of cancer tumor center's coordinates,
        sizes joined with binary classification task.

    Note
    ----
    y_true and y_pred tensors must have [None, 7] shapes;
    y_true[:, :3] and y_pred[:, :3] correspond to normalized (from [0, 1] interval)
    zyx coordinates of cancer tumor, while y_true[:, 3:6] and y_pred[:, 3:6]
    correspond to sizes of cancer tumor along zyx axes(also normalized),
    finally, y_true[:, 6] and y_pred[:, 6] represent whether cancer tumor presents
    or not in the current crop.
    """
    clf_true, clf_pred = y_true[:, 6], y_pred[:, 6]
    centers_true, centers_pred = y_true[:, :3], y_pred[:, :3]
    sizes_true, sizes_pred = y_true[:, 3:6], y_pred[:, 3:6]

    centers_loss = 0.5 * tf.reduce_sum((centers_true - centers_pred) ** 2, axis=1)
    sizes_loss = 0.5 * tf.reduce_sum((tf.sqrt(sizes_true) - tf.sqrt(sizes_pred)) ** 2, axis=1)
    clf_loss = 0.5 * (clf_true - clf_pred) ** 2

    loss = clf_loss + lambda_coords * clf_true * (centers_loss + sizes_loss)
    return tf.reduce_mean(loss)


def iou_3d(y_true, y_pred, epsilon=10e-7):
    """ Compute intersection over union in 3D case for input tensors.

    Parameters
    ----------
    y_true : tf.Tensor
        tensor containg true values for sizes of nodules and their centers.
    y_pred : tf.Tensor
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
        r_true = y_true[:, :3]
        r_pred = y_pred[:, :3]

        s_true = y_true[:, 3:6]
        s_pred = y_pred[:, 3:6]

        abs_r_diff = tf.abs(r_true - r_pred)
        abs_s_diff = tf.abs(s_true - s_pred)

        iou_tensor = tf.where(abs_r_diff < abs_s_diff, 2 * tf.minimum(s_true, s_pred),
                              tf.clip_by_value(s_true + s_pred - abs_r_diff, 0, 1))

        iou_tensor = (tf.reduce_prod(iou_tensor, axis=1)
                      / (tf.reduce_prod(s_true, axis=1)
                         + tf.reduce_prod(s_pred, axis=1) + tf_epsilon))
    return iou_tensor


def tiversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tiversky loss function.

    Parameters
    ----------
    y_true : tf.Tensor
        tensor containing target mask.
    y_pred : tf.Tensor
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
        tensor containing tiversky loss.
    """
    y_true = tf.contrib.layers.flatten(y_true)
    y_pred = tf.contrib.layers.flatten(y_pred)
    truepos = tf.reduce_sum(y_true * y_pred)
    fp_and_fn = (alpha * tf.reduce_sum(y_pred * (1 - y_true))
                 + beta * tf.reduce_sum((1 - y_pred) * y_true))

    return -(truepos + smooth) / (truepos + smooth + fp_and_fn)


def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    y_true : tf.Tensor
        tensor containing target mask.
    y_pred : tf.Tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    tf.Tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = tf.contrib.layers.flatten(y_true)
    y_pred = tf.contrib.layers.flatten(y_pred)
    truepos = tf.reduce_sum(y_true * y_pred)
    falsepos = tf.reduce_sum(y_pred) - truepos
    falseneg = tf.reduce_sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -tf.log(jaccard + smooth)
