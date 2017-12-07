""" Contains named expression classes"""


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline


class NamedExpression:
    """ Base class for a named expression """
    def __init__(self, name, copy=False):
        self.name = name
        self.copy = copy

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a named expression """
        if isinstance(self.name, NamedExpression):
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def set(self, value, batch=None, pipeline=None, model=None, mode='w'):
        """ Set a value to a named expression """
        if mode in ['a', 'append']:
            self.append(value, batch=batch, pipeline=pipeline, model=model)
        elif mode in ['e', 'extend']:
            self.extend(value, batch=batch, pipeline=pipeline, model=model)
        elif mode in ['u', 'update']:
            self.update(value, batch=batch, pipeline=pipeline, model=model)
        else:
            self.assign(value, batch=batch, pipeline=pipeline, model=model)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a named expression """
        raise NotImplementedError("assign should be implemented in child classes")

    def append(self, value, *args, **kwargs):
        """ Append a value to a named expression

        if a named expression is a dict or set, `update` is called, or `append` otherwise.

        See also
        --------
        list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
        dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update
        """
        var = self.get(*args, **kwargs)
        if isinstance(var, (set, dict)):
            var.update(value)
        else:
            var.append(value)

    def extend(self, value, *args, **kwargs):
        """ Extend a named expression with a new value
        (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) """
        self.get(*args, **kwargs).extend(value)

    def update(self, value, *args, **kwargs):
        """ Update a named expression with a new value
        (see dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        or set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update) """
        self.get(*args, **kwargs).update(value)


def eval_expr(expr, batch, pipeline=None, model=None):
    """ Evaluate a named expression recursively """
    args = dict(batch=batch, pipeline=pipeline, model=model)
    if isinstance(expr, NamedExpression):
        expr = expr.get(**args)
    elif isinstance(expr, (list, tuple)):
        _expr = []
        for val in expr:
            _expr.append(eval_expr(val, **args))
        expr = type(expr)(_expr)
    elif isinstance(expr, dict):
        _expr = type(expr)()
        for key, val in expr.items():
            key = eval_expr(key, **args)
            val = eval_expr(val, **args)
            _expr.update({key: val})
        expr = _expr
    return expr


class B(NamedExpression):
    """ Batch component name """
    def __init__(self, name=None, copy=True):
        super().__init__(name, copy)

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models B(%s)" % name)
        if name is None:
            return batch.deepcopy() if self.copy else batch
        return getattr(batch, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if name is not None:
            setattr(batch, name, value)


class C(NamedExpression):
    """ A pipeline config option """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}

        recursive_names = name.split('/')
        for n in recursive_names:
            config = config.get(n)
        return config

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        config[name] = value


class F(NamedExpression):
    """ A function, method or any other callable """
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name)
        self.args = args
        self.kwargs = kwargs

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value from a callable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        args = []
        if isinstance(batch, _DummyBatch) or batch is None:
            _pipeline = batch.pipeline if batch is not None else pipeline
            args += [_pipeline]
        else:
            args += [batch]
        if model is not None:
            args += [model]
        fargs = eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        fkwargs = eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)
        return name(*args, *fargs, **fkwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value with a callable is not supported")


class V(NamedExpression):
    """ Pipeline variable name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        return pipeline.get_variable(name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.set_variable(name, value)
