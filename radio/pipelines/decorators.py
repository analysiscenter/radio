""" Implementation of decorators used for creation of augmentation pipelines. """

from functools import partial, wraps
from itertools import product
import numpy as np

from ..dataset import Pipeline, F


def options_prod(**kwargs):
    """ Pipeline to generate batches with all combinations of given options used.

    Parameters
    ----------
    **kwargs : dict
        keys are options, values are lists, tuples or ndarrays with
        possible options' values.

    Returns
    -------
    callable
        decorator for pipeline generating function.
    """
    array_types = (tuple, list, np.ndarray)
    if not all(isinstance(v, array_types) for v in kwargs.values()):
        raise ValueError("For each parameter list of possible "
                         + " values must be provided")

    def decorator(pipeline_creator):
        """ Wrapper function for original pipeline generating function.

        Parameters
        ----------
        pipeline_creator : callable
            pipeline generating function.

        Returns
        -------
        callable
            new pipeline generating function.
        """
        @wraps(pipeline_creator)
        def wrapped(**creator_kwargs):
            """ Wrapped pipeline.

            Parameters
            ----------
            **creator_kwargs : dict
                parameters of original pipeline generating function.

            Returns
            -------
            Pipeline
            """
            pipelines = []

            for params in product(*kwargs.values()):
                pipeline = pipeline_creator(**creator_kwargs,
                                            **dict(zip(kwargs.keys(), params)))
                pipelines.append(pipeline)
            return (
                Pipeline()
                .apply(lambda b: b.concat([b >> pipe for pipe in pipelines]))
            )

        return wrapped

    return decorator


def repeat_pipeline(num_repeats=1):
    """ Decorator to create pipeline that repeats original pipeline given number of times.

    Parameters
    ----------
    num_repeats : int
        number of repeats.

    Returns
    -------
    callable
        decorator for pipeline generating function.
    """
    def decorator(pipeline_creator):
        @wraps(pipeline_creator)
        def wrapped(**kwargs):
            pipeline = pipeline_creator(**kwargs)
            return (
                Pipeline()
                .repeat_pipeline(pipeline, num_repeats=num_repeats)
            )

        return wrapped

    return decorator


def options_seq(**kwargs):
    """ Pipeline to generate batches with zipped combinations of parameters.

    Parameters
    ----------
    **kwargs : dict
        keys are options, values are lists, tuples or ndarrays with
        possible options' values.

    Returns
    -------
    callable
        decorator for pipeline generating function.
    """
    array_types = (tuple, list, np.ndarray)
    if not all(isinstance(v, array_types) for v in kwargs.values()):
        raise ValueError("For each parameter list of possible "
                         + " values must be provided")

    def decorator(pipeline_creator):
        """ Wrapper function for original pipeline generating function.

        Parameters
        ----------
        pipeline_creator : callable
            pipeline generating function.

        Returns
        -------
        callable
            new pipeline generating function.
        """
        @wraps(pipeline_creator)
        def wrapped(**creator_kwargs):
            """ Wrapped pipeline.

            Parameters
            ----------
            **creator_kwargs : dict
                parameters of original pipeline generating function.

            Returns
            -------
            Pipeline
            """
            pipelines = []
            for params in zip(*kwargs.values()):
                pipeline = pipeline_creator(**creator_kwargs,
                                            **dict(zip(kwargs.keys(), params)))
                pipelines.append(pipeline)
            return (
                Pipeline()
                .apply(lambda b: b.concat([b >> pipe for pipe in pipelines]))
            )

        return wrapped

    return decorator
