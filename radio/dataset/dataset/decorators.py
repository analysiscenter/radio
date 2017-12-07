""" Pipeline decorators """
import os
import traceback
import threading
import concurrent.futures as cf
import asyncio
import functools
import logging

from .named_expr import _DummyBatch, eval_expr


def _workers_count():
    cpu_count = 0
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return cpu_count * 4


_MODEL_MODES = ['global', 'static', 'dynamic']


class ModelDirectory:
    """ Directory of model definition methods in Batch classes """
    models = dict(zip(_MODEL_MODES, (dict() for _ in range(len(_MODEL_MODES)))))

    @staticmethod
    def print(pipeline=None):
        """ Print a model directory """
        for mode in ModelDirectory.models:
            print(mode)
            for pipe in ModelDirectory.models[mode]:
                print("  ", pipe)
                for amodel in ModelDirectory.models[mode][pipe]:
                    print("    ", amodel)
        for amodel in pipeline.models:
            print(amodel)

    @staticmethod
    def get_method_fullname(method):
        """ Return a method name in the format module_name.class_name.func_name """
        if callable(method):
            return method.__module__ + '.' + method.__name__
        return method

    @staticmethod
    def add_model(method_spec, model_spec):
        """ Add a model specification into the model directory """
        mode, model_method, pipeline = method_spec['mode'], method_spec['method'], method_spec['pipeline']
        if mode == 'global' or pipeline is None:
            if pipeline not in ModelDirectory.models[mode]:
                ModelDirectory.models[mode][pipeline] = dict()
            ModelDirectory.models[mode][pipeline].update({model_method: model_spec})
        else:
            pipeline.models.update({model_method: model_spec})

    @staticmethod
    def equal_names(model_name, model_ref):
        """ Check if model_name equals a full model name stored in model_ref """
        name = ModelDirectory.get_method_fullname(model_name)
        return model_ref[-len(name):] == name

    @staticmethod
    def find_model_method_by_name(model_name, pipeline=None, modes=None):
        """ Search a model method by its name """
        modes = modes or _MODEL_MODES
        model_dicts = []
        if pipeline is not None:
            model_dicts.append(pipeline.models)
        for mode in modes:
            if None in ModelDirectory.models[mode]:
                mode_models = ModelDirectory.models[mode][None]
                model_dicts.append(mode_models)
        models_with_same_name = []
        for mode_models in model_dicts:
            models_with_same_name += [model_method for model_method in mode_models
                                      if ModelDirectory.equal_names(model_name, model_method.method_spec['name'])]
        return models_with_same_name if len(models_with_same_name) > 0 else None

    @staticmethod
    def find_model_by_name(name, pipeline=None, only_first=False, modes=None):
        """ Search a model by its name """
        all_model_methods = ModelDirectory.find_model_method_by_name(name, pipeline=pipeline, modes=modes) or []

        method_specs = [model_method.method_spec for model_method in all_model_methods
                        if hasattr(model_method, 'method_spec')]
        model_specs = [ModelDirectory.get_model(method_spec, pipeline) for method_spec in method_specs]

        if len(model_specs) == 0:
            return None
        elif len(model_specs) == 1 or only_first:
            return model_specs[0]
        return model_specs

    @staticmethod
    def model_exists(method_spec, pipeline=None):
        """ Check if a model specification exists in the model directory """
        mode, model_method, _pipeline = method_spec['mode'], method_spec['method'], method_spec['pipeline']
        pipeline = pipeline if pipeline is not None else _pipeline
        pipeline = None if mode == 'global' else pipeline
        return pipeline in ModelDirectory.models[mode] and model_method in ModelDirectory.models[mode][pipeline] or \
               model_method in pipeline.models and pipeline.models[model_method] is not None

    @staticmethod
    def get_model(method_spec, pipeline=None):
        """ Return a model specification for a given model method

        Returns
        -------
        a model specification or a list of model specifications

        Raises
        ------
        `ValueError` if a model has not been found
        """
        mode, model_method, _pipeline = method_spec['mode'], method_spec['method'], method_spec['pipeline']
        pipeline = pipeline if pipeline is not None else _pipeline
        pipeline = None if mode == 'global' else pipeline

        if pipeline in ModelDirectory.models[mode] and model_method in ModelDirectory.models[mode][pipeline]:
            return ModelDirectory.models[mode][pipeline][model_method]
        elif model_method in pipeline.models:
            return pipeline.models[model_method]

    @staticmethod
    def get_model_by_name(model_name, batch=None, pipeline=None):
        """ Return a model specification given its name

        Parameters
        ----------
        model_name: str - a name of the model
                    callable - a method or a function with a model definition
        batch - an instance of the batch class where to look for a model or None
        pipeline - a pipeline where to look for a model or None

        Returns
        -------
        a model specification or a list of model specifications

        Raises
        ------
        `ValueError` if a model has not been found
        """
        pipeline = pipeline if pipeline is not None else batch.pipeline
        model_spec = ModelDirectory.find_model_by_name(model_name, pipeline, only_first=True)
        if model_spec is None:
            if batch is None:
                # when called outside of a batch or a pipeline
                if pipeline is None:
                    raise ValueError("Model '%s' not found" % model_name)
                else:
                    raise ValueError("Model '%s' not found in the pipeline %s" % (model_name, pipeline))
            else:
                if callable(model_name):
                    method = functools.partial(model_name, batch)
                else:
                    # if a model is defined in a model method within a batch class
                    if hasattr(batch, model_name):
                        method = getattr(batch, model_name)
                    else:
                        model_method = ModelDirectory.find_model_method_by_name(model_name, pipeline)
                        if model_method is None:
                            raise ValueError("Model '%s' not found neither in the batch class %s,"
                                             % (model_name, batch.__class__.__name__) +
                                             " nor in the pipeline %s" % pipeline)
                        else:
                            # a model method is supposed to be in a Batch class, so batch serves as self
                            method = functools.partial(model_method[0], batch)
                model_spec = method()
        return model_spec

    @staticmethod
    def del_model(method_spec):
        """ Remove a model specification from the model directory """
        mode, model_method, pipeline = method_spec['mode'], method_spec['method'], method_spec['pipeline']
        ModelDirectory.models[mode][pipeline].pop(model_method, None)

    @staticmethod
    def delete_all_models(pipeline):
        """ Remove all models created in a pipeline """
        model_dicts = []
        for mode in _MODEL_MODES[1:]:
            if pipeline in ModelDirectory.models[mode]:
                mode_models = ModelDirectory.models[mode][pipeline]
                model_dicts.append(mode_models)
        for mode_models in model_dicts:
            for one_model in mode_models:
                method_spec = {**one_model.method_spec, **dict(pipeline=pipeline)}
                ModelDirectory.del_model(method_spec)

    @staticmethod
    def init_model(mode, model_class=None, name=None, pipeline=None, config=None):
        """ Initialize a static or dynamic model in a pipeline

        Parameters
        ----------
        mode : str - 'static' or 'dynamic'
        model_class : class - a model class to instantiate a model
        name : str - a short name for the model
        pipeline - a pipeline to link a model to
        config : dict - model configurations parameters, where each key and value could be named expressions
            - B('name') - a batch class attribute or component name
            - V('name') - a pipeline variable name
            - C('name') - a pipeline config option
            - F(name) - a callable which takes a batch for dynamic models or a pipeline for static models
            These expressions will be substituted by their actual values.
            All other value will be used "as is".
        """
        if model_class is not None:
            name = name or model_class.__name__
            init_config = config

            def _model_definition_maker():
                def _model_definition_method(batch_or_pipe, config=None):
                    def _calc_expr(val):
                        if mode == 'dynamic':
                            args = dict(batch=batch_or_pipe)
                        else:
                            dummy_batch = _DummyBatch(batch_or_pipe)
                            args = dict(batch=dummy_batch)
                        out = eval_expr(val, **args)
                        return out

                    global_config = config or dict()
                    local_config = init_config or dict()

                    global_config.update(_calc_expr(global_config))
                    global_config.update(_calc_expr(local_config))
                    return model_class(name=name, config=global_config)


                _model_definition_method.__name__ = name
                return _model_definition_method

            model_method = model(mode=mode, pipeline=pipeline)(_model_definition_maker())
            pipeline.models.update({model_method: None})

        if mode == 'static':
            model_methods = ModelDirectory.find_model_method_by_name(name, pipeline, ['static'])
            if model_methods is None:
                model_methods = ModelDirectory.find_model_method_by_name(name, None, ['static'])
                if model_methods is None:
                    raise ValueError("Model '%s' not found in the pipeline %s" % (name, pipeline))
            if len(model_methods) > 1:
                raise ValueError("There are several models with the name '%s' in the pipeline %s" \
                                 % (name, pipeline))
            # a model method is supposed to be in a Batch class, so dummy_batch is a fake self
            dummy_batch = _DummyBatch(pipeline)
            _ = model_methods[0](dummy_batch)


    @staticmethod
    def import_model(name, model_spec, pipeline):
        """ Import a model into a pipeline """
        def _model_definition_maker():
            def _model_definition_method(*args, **kwargs):
                _ = args, kwargs
                return model_spec
            _model_definition_method.__name__ = name
            return _model_definition_method

        model_method = model(mode='dynamic', pipeline=pipeline)(_model_definition_maker())
        dummy_batch = _DummyBatch(pipeline)
        model_method(dummy_batch)

    @staticmethod
    def import_model_from(name, from_pipeline, to_pipeline):
        """ Import a model from another pipeline """
        model_methods = ModelDirectory.find_model_method_by_name(name, from_pipeline)
        if model_methods is None:
            raise RuntimeError("Model '%s' does not exist in the pipeline %s" % (name, from_pipeline))
        if len(model_methods) > 1:
            raise RuntimeError("There are a few models with the name '%s' in the pipeline %s"
                               % (name, from_pipeline))

        model_method = model_methods[0]
        if hasattr(model_method, 'method_spec'):
            method_spec = model_method.method_spec
        else:
            raise RuntimeError("Method %s is not decorated with @model" % name)

        model_spec = ModelDirectory.get_model(method_spec, from_pipeline)
        method_spec = {**method_spec, **dict(mode='dynamic', pipeline=to_pipeline)}
        ModelDirectory.add_model(method_spec, model_spec)


def model(mode='global', pipeline=None):
    """ Decorator for model methods

    Examples
    --------
    @model()
    def global_model():
        ...
        return my_model

    @model(mode='static')
    def static_model(config=None):
        ...
        return my_model

    @model(mode='dynamic')
    def some_model(self, config=None):
        ...
        return my_model
    """
    def _model_decorator(method):

        _pipeline_model_lock = threading.Lock()

        def _get_method_spec(pipeline=None):
            return dict(mode=mode, method=_model_wrapper, pipeline=pipeline,
                        name=ModelDirectory.get_method_fullname(method))

        def _add_model(method_spec, model_spec):
            ModelDirectory.add_model(method_spec, model_spec)

        @functools.wraps(method)
        def _model_wrapper(batch, config=None):
            if mode == 'global':
                pipeline = None
            elif mode in ['static', 'dynamic']:
                pipeline = batch.pipeline
                method_spec = _get_method_spec()
                method_spec['pipeline'] = pipeline

                if not ModelDirectory.model_exists(method_spec):
                    with _pipeline_model_lock:
                        if not ModelDirectory.model_exists(method_spec):
                            if pipeline is not None:
                                full_config = pipeline.config
                                full_model_name = ModelDirectory.get_method_fullname(method)
                                if full_config is not None:
                                    model_names = [model_key for model_key in full_config
                                                   if ModelDirectory.equal_names(model_key, full_model_name)]
                                    if len(model_names) > 1:
                                        raise ValueError("Ambigous config contains several keys " +
                                                         "with similar names", model_names)
                                    elif len(model_names) == 1:
                                        config = full_config[model_names[0]] or dict()

                            args = (pipeline,) if mode == 'static' else (batch,)
                            args = args if config is None else args + (config,)
                            model_spec = method(*args)
                            _add_model(method_spec, model_spec)
            else:
                raise ValueError("Unknown mode", mode)

            model_spec = ModelDirectory.get_model(method_spec, pipeline)
            return model_spec


        method_spec = _get_method_spec(pipeline)
        if mode == 'global':
            model_spec = method()
            _add_model(method_spec, model_spec)
        elif mode == 'static':
            _add_model(method_spec, dict())
        _model_wrapper.model_method = _model_wrapper
        _model_wrapper.method_spec = method_spec
        method.method_spec = method_spec
        return _model_wrapper
    return _model_decorator



def _make_action_wrapper_with_args(model=None, use_lock=None):    # pylint: disable=redefined-outer-name
    return functools.partial(_make_action_wrapper, _model_name=model, _use_lock=use_lock)

def _make_action_wrapper(action_method, _model_name=None, _use_lock=None):
    @functools.wraps(action_method)
    def _action_wrapper(action_self, *args, **kwargs):
        """ Call the action method """
        if _use_lock is not None:
            if action_self.pipeline is not None:
                if isinstance(_use_lock, bool):
                    _lock_name = '#_lock_' + action_method.__name__
                else:
                    _lock_name = _use_lock
                if not action_self.pipeline.has_variable(_lock_name):
                    action_self.pipeline.init_variable(_lock_name, threading.Lock())
                action_self.pipeline.get_variable(_lock_name).acquire()

        if _model_name is None:
            _res = action_method(action_self, *args, **kwargs)
        else:
            _model_spec = ModelDirectory.get_model_by_name(_model_name, pipeline=action_self.pipeline,
                                                           batch=action_self)
            _res = action_method(action_self, _model_spec, *args, **kwargs)

        if _use_lock is not None:
            if action_self.pipeline is not None:
                action_self.pipeline.get_variable(_lock_name).release()

        return _res

    _action_wrapper.action = dict(method=action_method, use_lock=_use_lock)
    return _action_wrapper

def action(*args, **kwargs):
    """ Decorator for action methods in :class:`~dataset.Batch` classes

    Examples ::
    --------

    .. code-block:: python

        @action
        def some_action(self, arg1, arg2):
            ...

        @action(model='some_model')
        def train_model(self, model, another_arg):
            ...

        @action(use_lock=True)
        def critical_section(self, some_arg, another_arg):
            ...

        @action(use_lock='lock_name')
        def another_critical_section(self, some_arg, another_arg):
            ...
    """
    if len(args) == 1 and callable(args[0]):
        # action without arguments
        return _make_action_wrapper(action_method=args[0])
    # action with arguments
    return _make_action_wrapper_with_args(*args, **kwargs)


def any_action_failed(results):
    """ Return `True` if some parallelized invocations threw exceptions """
    return any(isinstance(res, Exception) for res in results)

def inbatch_parallel(init, post=None, target='threads', **dec_kwargs):
    """ Decorator for parallel methods in :class:`~dataset.Batch` classes"""
    if target not in ['nogil', 'threads', 'mpc', 'async', 'for', 't', 'm', 'a', 'f']:
        raise ValueError("target should be one of 'threads', 'mpc', 'async', 'for'")

    def inbatch_parallel_decorator(method):
        """ Return a decorator which run a method in parallel """
        def _check_functions(self):
            """ Check dcorator's `init` and `post` parameters """
            if init is None:
                raise ValueError("init cannot be None")
            else:
                try:
                    init_fn = getattr(self, init)
                except AttributeError:
                    raise ValueError("init should refer to a method or property of the class", type(self).__name__,
                                     "returning the list of arguments")
            if post is not None:
                try:
                    post_fn = getattr(self, post)
                except AttributeError:
                    raise ValueError("post should refer to a method of the class", type(self).__name__)
                if not callable(post_fn):
                    raise ValueError("post should refer to a method of the class", type(self).__name__)
            else:
                post_fn = None
            return init_fn, post_fn

        def _call_init_fn(init_fn, args, kwargs):
            if callable(init_fn):
                return init_fn(*args, **kwargs)
            return init_fn

        def _call_post_fn(self, post_fn, futures, args, kwargs):
            all_results = []
            for future in futures:
                try:
                    if isinstance(future, (cf.Future, asyncio.Task)):
                        result = future.result()
                    else:
                        result = future
                except Exception as exce:  # pylint: disable=broad-except
                    result = exce
                finally:
                    all_results += [result]

            if post_fn is None:
                if any_action_failed(all_results):
                    all_errors = [error for error in all_results if isinstance(error, Exception)]
                    print(all_errors)
                    traceback.print_tb(all_errors[0].__traceback__)
                return self
            else:
                return post_fn(all_results, *args, **kwargs)

        def _make_args(init_args, args, kwargs):
            """ Make args, kwargs tuple """
            if isinstance(init_args, tuple) and len(init_args) == 2:
                margs, mkwargs = init_args
            elif isinstance(init_args, dict):
                margs = list()
                mkwargs = init_args
            else:
                margs = init_args
                mkwargs = dict()
            margs = margs if isinstance(margs, (list, tuple)) else [margs]
            if len(args) > 0:
                margs = list(margs) + list(args)
            if len(kwargs) > 0:
                mkwargs.update(kwargs)
            return margs, mkwargs

        def wrap_with_threads(self, args, kwargs, nogil=False):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                if nogil:
                    nogil_fn = method(self, *args, **kwargs)
                full_kwargs = {**dec_kwargs, **kwargs}
                for arg in _call_init_fn(init_fn, args, full_kwargs):
                    margs, mkwargs = _make_args(arg, args, kwargs)
                    if nogil:
                        one_ft = executor.submit(nogil_fn, *margs, **mkwargs)
                    else:
                        one_ft = executor.submit(method, self, *margs, **mkwargs)
                    futures.append(one_ft)

                timeout = kwargs.get('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_mpc(self, args, kwargs):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                mpc_func = method(self, *args, **kwargs)
                full_kwargs = {**dec_kwargs, **kwargs}
                for arg in _call_init_fn(init_fn, args, full_kwargs):
                    margs, mkwargs = _make_args(arg, args, kwargs)
                    one_ft = executor.submit(mpc_func, *margs, **mkwargs)
                    futures.append(one_ft)

                timeout = kwargs.pop('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_async(self, args, kwargs):
            """ Run a method in parallel with async / await """
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # this is a new thread where there is no loop
                loop = kwargs.get('loop', None)
                asyncio.set_event_loop(loop)
            else:
                loop = kwargs.get('loop', loop)

            init_fn, post_fn = _check_functions(self)

            futures = []
            full_kwargs = {**dec_kwargs, **kwargs}
            for arg in _call_init_fn(init_fn, args, full_kwargs):
                margs, mkwargs = _make_args(arg, args, kwargs)
                futures.append(asyncio.ensure_future(method(self, *margs, **mkwargs)))

            loop.run_until_complete(asyncio.gather(*futures, loop=loop, return_exceptions=True))

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_for(self, args, kwargs):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            _ = kwargs.pop('n_workers', _workers_count())
            futures = []
            full_kwargs = {**dec_kwargs, **kwargs}
            for arg in _call_init_fn(init_fn, args, full_kwargs):
                margs, mkwargs = _make_args(arg, args, kwargs)
                try:
                    one_ft = method(self, *margs, **mkwargs)
                    if callable(one_ft):
                        one_ft = one_ft(*margs, **mkwargs)
                except Exception as e:   # pylint: disable=broad-except
                    one_ft = e
                futures.append(one_ft)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        @functools.wraps(method)
        def wrapped_method(self, *args, **kwargs):
            """ Wrap a method in a required parallel engine """
            if asyncio.iscoroutinefunction(method) or target == 'async':
                return wrap_with_async(self, args, kwargs)
            if target in ['threads', 't']:
                return wrap_with_threads(self, args, kwargs)
            elif target == 'nogil':
                return wrap_with_threads(self, args, kwargs, nogil=True)
            elif target in ['mpc', 'm']:
                return wrap_with_mpc(self, args, kwargs)
            elif target in ['for', 'f']:
                return wrap_with_for(self, args, kwargs)
            raise ValueError('Wrong parallelization target:', target)
        return wrapped_method
    return inbatch_parallel_decorator


parallel = inbatch_parallel  # pylint: disable=invalid-name

def njit(nogil=True):
    """ Fake njit decorator to use when numba is not installed """
    _ = nogil
    def njit_fake_decorator(method):
        """ Return a decorator """
        @functools.wraps(method)
        def wrapped_method(*args, **kwargs):
            """ Log warning that numba is not installed which causes preformance degradation """
            logging.warning('numba is not installed. This causes a severe performance degradation for method %s',
                            method.__name__)
            return method(*args, **kwargs)
        return wrapped_method
    return njit_fake_decorator
