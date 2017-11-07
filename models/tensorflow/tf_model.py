# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=not-context-manager
""" Contains base class for all tensorflow models. """

from ...dataset.dataset.models.tf import TFModel


class TFModelCT(TFModel):
    """ Base class for all tensorflow models.

    This class inherits TFModel class from dataset submodule and
    extends it with metrics accumulating methods. Also
    train and predict methods were overloaded:
    train method gets 'x' and 'y',
    while predict gets only 'x' as arguments instead of 'feed_dict'
    and 'fetches' as it was in parent class. It's simplifies interface
    and makes TFModel3D compatible with KerasModel interface.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def train(self, x=None, y=None, **kwargs):
        """ Train model with data provided.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            numpy array that will be fed into tf.placeholder that can be accessed
            by 'x' attribute of 'self', typically input of neural network.
        y : ndarray(batch_size, ...)
            numpy array that will be fed into tf.placeholder that can be accessed
            by 'y' attribute of 'self'.

        Returns
        -------
        ndarray(batch_size, ...)
            predicted output.
        """
        _fetches = ('y_pred', )
        train_output = super().train(_fetches, {'x': x, 'y': y})
        return train_output

    def predict(self, x=None, **kargs):
        """ Predict model on data provided.

        Parameters
        ----------
        x : ndarray(batch_size, ....)
            numpy array that will be fed into tf.placeholder that can be accessed
            by 'x' attribute of 'self', typically input of neural network.

        Returns
        -------
        ndarray(batch_size, ...)
            predicted output.
        """
        predictions = super().predict(fetches=None, feed_dict={'x': x})
        return predictions
