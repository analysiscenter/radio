""" Contains base class for both keras and tensorflow model. """


class BaseModel(object):
    """ Base model class for both keras and tensorflow models. """

    def __init__(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        """ Complile model. """
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """ Load model. """
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """ Save model. """
        raise NotImplementedError

    def train_on_batch(self, x, y_true, **kwargs):
        """ Train model on batch. """
        raise NotImplementedError

    def test_on_batch(self, x, y_true, **kwargs):
        """ Test model on batch. """
        raise NotImplementedError

    def predict_on_batch(self, x, **kwargs):
        """ Predict on batch. """
        raise NotImplementedError
