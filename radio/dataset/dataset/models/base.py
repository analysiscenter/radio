""" Contains a base model class"""

class BaseModel:
    """ Base class for all models

    Attributes
    ----------
    name : str
        a model name
    config : dict
        configuration parameters

    Notes
    -----

    **Configuration**:

    * build : bool
        whether to build a model by calling `self.build()`. Default is True.
    * load : dict
        parameters for model loading. If present, a model will be loaded
        by calling `self.load(**config['load'])`.

    """
    def __init__(self, name=None, config=None, *args, **kwargs):
        self.config = config or {}
        self.name = name or self.__class__.__name__
        if self.get('build', self.config, True):
            self.build(*args, **kwargs)
        load = self.get('load', self.config, False)
        if load:
            self.load(**load)

    @classmethod
    def pop(cls, variables, config=None):
        """ Return variables and remove them from config"""
        return cls.get(variables, config, pop=True)

    @classmethod
    def get(cls, variables, config=None, default=None, pop=False):
        """ Return variables from config """
        unpack = False
        if not isinstance(variables, (list, tuple)):
            variables = list([variables])
            unpack = True

        ret_vars = []
        for variable in variables:
            _config = config
            if '/' in variable:
                var = variable.split('/')
                prefix = var[:-1]
                var_name = var[-1]
            else:
                prefix = []
                var_name = variable

            for p in prefix:
                if p in _config:
                    _config = _config[p]
                else:
                    _config = None
                    break
            if _config:
                if pop:
                    val = _config.pop(var_name)
                else:
                    val = _config.get(var_name, default)
            else:
                raise KeyError("Key '%s' not found" % variable)

            ret_vars.append(val)

        if unpack:
            ret_vars = ret_vars[0]
        else:
            ret_vars = tuple(ret_vars)
        return ret_vars

    @classmethod
    def put(cls, variable, value, config):
        """ Put a new variable into config """
        if '/' in variable:
            var = variable.strip('/').split('/')
            prefix = var[:-1]
            var_name = var[-1]
        else:
            prefix = []
            var_name = variable

        for p in prefix:
            if p not in config:
                config[p] = dict()
            config = config[p]
        config[var_name] = value

    def _make_inputs(self, names=None, config=None):
        """ Make model input data using config

        Parameters
        ----------
        names : a sequence of str - names for input variables

        Returns
        -------
        None or dict - where key is a variable name and a value is a corresponding variable after configuration
        """
        _ = names, config
        return None

    def build(self, *args, **kwargs):
        """ Define the model """
        _ = self, args, kwargs

    def load(self, *args, **kwargs):
        """ Load the model """
        _ = self, args, kwargs

    def save(self, *args, **kwargs):
        """ Save the model """
        _ = self, args, kwargs

    def train(self, *args, **kwargs):
        """ Train the model """
        _ = self, args, kwargs

    def predict(self, *args, **kwargs):
        """ Make a prediction using the model  """
        _ = self, args, kwargs
