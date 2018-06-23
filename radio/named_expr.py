""" Contains named expressions from dataset module extended with arithmetics. """

import numpy as np
from .dataset.dataset.named_expr import NamedExpression, _DummyBatch


INT_TYPES = (int, np.int, np.int8, np.int16, np.int32, np.int64)
FLOAT_TYPES = (float, np.float, np.float16, np.float32, np.float64)


class NamedExpressionWithArithmetics(NamedExpression):
    """ Base class for a named expression """
    def __init__(self, name, copy=False):
        self.name = name
        self.copy = copy

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a named expression """
        if isinstance(self.name, NamedExpression):
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        if callable(self.name):
            return self.name(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def set(self, value, batch=None, pipeline=None, model=None, mode='w'):
        """ Set a value to a named expression """
        value = self.eval_expr(value, batch=batch, pipeline=pipeline, model=model)
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
        self.get(*args, **kwargs).update(value)

    def __repr__(self):
        return type(self).__name__ + '(' + str(self.name) + ')'

    @classmethod
    def eval_expr(cls, expr, batch=None, pipeline=None, model=None):
        """ Evaluate a named expression recursively """
        if batch is None:
            batch = _DummyBatch(pipeline)
        args = dict(batch=batch, pipeline=pipeline, model=model)

        if isinstance(expr, cls):
            _expr = expr.get(**args)
            if isinstance(_expr, cls):
                expr = cls.eval_expr(_expr, **args)
            else:
                expr = _expr
        elif isinstance(expr, (list, tuple)):
            _expr = []
            for val in expr:
                _expr.append(cls.eval_expr(val, **args))
            expr = type(expr)(_expr)
        elif isinstance(expr, dict):
            _expr = type(expr)()
            for key, val in expr.items():
                key = cls.eval_expr(key, **args)
                val = cls.eval_expr(val, **args)
                _expr.update({key: val})
            expr = _expr
        return expr

    def __pos__(self):
        return self

    def __neg__(self):
        def exp_value(batch=None, pipeline=None, model=None):
            return -self.get(batch, pipeline, model)
        return NamedExpressionWithArithmetics(exp_value)

    def __add__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) + other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) + other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __radd__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) + other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) + other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __mul__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) * other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) * other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __rmul__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) * other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) * other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __sub__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) - other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) - other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __rsub__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return other - self.get(batch, pipeline, model)
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return  other.get(batch, pipeline, model) - self.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __floordiv__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) // other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) // other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __rfloordiv__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return other // self.get(batch, pipeline, model)
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return other.get(batch, pipeline, model) // self.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __truediv__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) / other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) / other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __rtruediv__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return other / self.get(batch, pipeline, model)
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return other.get(batch, pipeline, model) / self.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __mod__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) % other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) % other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __rmod__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return other % self.get(batch, pipeline, model)
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return other.get(batch, pipeline, model) % self.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __pow__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) ** other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) ** other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __rpow__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return other ** self.get(batch, pipeline, model)
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return other.get(batch, pipeline, model) ** self.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __lt__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) < other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) < other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def ___le__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) <= other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) <= other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __eq__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) == other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) == other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __ne__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) != other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) != other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __gt__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) > other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) > other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)

    def __ge__(self, other):
        if isinstance(other, INT_TYPES + FLOAT_TYPES + (np.ndarray, )):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) >= other
        elif isinstance(other, NamedExpression):
            def exp_value(batch=None, pipeline=None, model=None):
                return self.get(batch, pipeline, model) >= other.get(batch, pipeline, model)
        else:
            raise ValueError("Operands must be be ints, floats or NamedExpressions.")
        return NamedExpressionWithArithmetics(exp_value)


class B(NamedExpressionWithArithmetics):
    def __init__(self, name=None, copy=True):
        super().__init__(name, copy)

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models: B('%s')" % name)
        if name is None:
            return batch.deepcopy() if self.copy else batch
        return getattr(batch, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if name is not None:
            setattr(batch, name, value)


class C(NamedExpressionWithArithmetics):

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


class F(NamedExpressionWithArithmetics):
    def __init__(self, name=None, *args, _pass=True, **kwargs):
        super().__init__(name)
        self.args = args
        self.kwargs = kwargs
        self._pass = _pass

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value from a callable """
        if isinstance(self.name, NamedExpression):
            name = self.name.get(batch=batch, pipeline=pipeline, model=model)
        else:
            name = self.name
        args = []
        if self._pass:
            if isinstance(batch, _DummyBatch) or batch is None:
                _pipeline = batch.pipeline if batch is not None else pipeline
                args += [_pipeline]
            else:
                args += [batch]
            if model is not None:
                args += [model]
        fargs = self.eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        fkwargs = self.eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)
        return name(*args, *fargs, **fkwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value with a callable is not supported")


class V(NamedExpressionWithArithmetics):

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        value = pipeline.get_variable(name)
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.assign_variable(name, value, batch=batch)
