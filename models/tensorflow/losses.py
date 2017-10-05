""" Contains lossed used in tensorflow models. """

import tensorflow as tf


def log_loss(y_true, y_pred, epsilon=10e-7):
    """ Log loss on input tensors.

    Args:
    - y_true: tf.Tensor, contains true labels;
    - y_pred: tf.Tensor, contains predicted logits;
    - epsilon: float, epsilon to avoid computing log(0);

    Returns:
    - tf.Variable, log loss on input tensors;
    """
    return tf.reduce_mean(y_true * tf.log(y_pred + epsilon)
                          + (1 - y_true) * tf.log(1 - y_pred + epsilon))


def reg_l2_loss(y_true, y_pred, lambda_coords):
    """ L2 loss for prediction of cancer tumor's centers, sizes joined with binary classification task.

    Args:
    - y_true: tf.Tensor, contains true values for sizes of nodules, their centers
    and classes of crop;
    - y_pred: tf.Tensor, contains predicted values for sizes of nodules, their centers
    and classes of crop;

    Returns:
    - tf.Variable, l2 loss for regression of cancer tumor center's coordinates,
    sizes joined with binary classification task.

    NOTE: y_true and y_pred tensors must have [None, 7] shapes;
    y_true[:, :3] and y_pred[:, :3] correspond to normalized (from [0, 1] interval)
    zyx coordinates of cancer tumor, while y_true[:, 3:6] and y_pred[:, 3:6]
    correspond to sizes of cancer tumor along zyx axes(also normalized),
    finally, y_true[:, 6] and y_pred[:, 6] represent whether cancer tumor presents
    or not in the current crop;
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

    Args:
    - y_true: tf.Tensor, contains true values for sizes of nodules and their centers;
    - y_pred: tf.Tensor, contains predicted values for sizes of nodules and their centers;

    Returns:
    - tf.Variable containing intersection over union computed on input tensors.
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


def dice_coef(y_true, y_pred, smooth=1e-7):
    """ Dice coefficient function implemente via tensorflow.

    Args:
    - y_true: tf.Tensor with target masks;
    - y_pred: tf.Tensor with predicted masks;

    Returns:
    - tf.Tensor with dice coefficient value;
    """
    y_true_f = tf.contrib.layers.flatten(y_true)
    y_pred_f = tf.contrib.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f)
                                             + tf.reduce_sum(y_pred_f) + smooth)
    return answer


def tiversky_coef(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tiversky coefficient.

    Args:
    - y_true: tf.Tensor with target masks;
    - y_pred: tf.Tensor with predicted masks;

    Returns:
    - tf.Tensor containing tiverky coefficient;
    """
    y_true = tf.contrib.layers.flatten(y_true)
    y_pred = tf.contrib.layers.flatten(y_pred)
    truepos = tf.reduce_sum(y_true * y_pred)
    fp_and_fn = (alpha * tf.reduce_sum(y_pred * (1 - y_true))
                 + beta * tf.reduce_sum((1 - y_pred) * y_true))

    return (truepos + smooth) / (truepos + smooth + fp_and_fn)


def jaccard_coef(y_true, y_pred, smooth=1e-10):
    """ Jaccard coefficient.

    Args:
    - y_true: tf.Tensor, actual pixel-by-pixel values for all classes;
    - y_pred: tf.Tensor, predicted pixel-by-pixel values for all classes;

    Returns:
    - tf.Tensor, jaccard score across all classes;
    """
    y_true = tf.contrib.layers.flatten(y_true)
    y_pred = tf.contrib.layers.flatten(y_pred)
    truepos = tf.reduce_sum(y_true * y_pred)
    falsepos = tf.reduce_sum(y_pred) - truepos
    falseneg = tf.reduce_sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return jaccard


def tiversky_loss(y_true, y_pred):
    """ Tiversky loss function.

    Args:
    - y_true: tf.Tensor containing target mask;
    - y_pred: tf.Tensor containing predicted mask;

    Returns:
    - tf.Tensor containing tiversky loss;
    """
    return -tiversky_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    """ Loss function base on dice coefficient.

    Args:
    - y_true: tf.Tensor containing target mask;
    - y_pred: tf.Tensor containing predicted mask;

    Returns:
    - tf.Tensor containing tiversky loss;
    """
    return -tiversky_coef(y_true, y_pred)


def jaccard_coef_logloss(y_true, y_pred):
    """ Loss function based on jaccard coefficient.

    Args:
    - y_true: tf.Tensor containing target mask;
    - y_pred: tf.Tensor containing predicted mask;

    Returns:
    - tf.Tensor with negative logarithm of jaccard coefficient;
    """
    return -tf.log(jaccard_coef(y_true, y_pred))
