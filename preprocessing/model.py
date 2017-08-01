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

    def fit(self, *args, **kwargs):
        """ Fit model. """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """ Get model prediction. """
        raise NotImplementedError
