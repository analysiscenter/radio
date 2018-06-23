import random
import numpy as np
from numba import njit


@njit(nogil=True)
def salt_and_pepper_numba(image, value, p):
    mask = np.random.binomial(1, p, image.shape)
    return image * (1 - mask) + mask * value


@njit(nogil=True)
def coarse_salt_and_pepper_numba(image, window, p, insert_value, coarse):
    out_image = np.empty_like(image)
    shape = np.array(image.shape, dtype=np.int64)
    num_iters = np.ceil(shape / window).astype(np.int64)
    for i in range(num_iters[0]):
        for j in range(num_iters[1]):
            for k in range(num_iters[2]):
                flag = random.random() < p
                image_crop = image[i * window[0]: (i + 1) * window[0],
                                   j * window[1]: (j + 1) * window[1],
                                   k * window[2]: (k + 1) * window[2]]

                if not flag:
                    value = image_crop
                elif coarse:
                    value = salt_and_pepper_numba(image_crop, insert_value, p)
                else:
                    value = np.ones_like(image_crop) * insert_value

                out_image[i * window[0]: (i + 1) * window[0],
                          j * window[1]: (j + 1) * window[1],
                          k * window[2]: (k + 1) * window[2]] = value
    return out_image


def salt_and_pepper(image, p=0.1, size_percent=None,
                    size=None, insert_value=0, coarse=False):

    if size is None and size_percent is None:
        raise ValueError("At least one of 'size', 'size_percent' "
                         + "arguments must be not None.")

    elif size is not None and size_percent is not None:
        raise ValueError("Only one of 'size', 'size_percent' " +
                         "arguments must be provided.")

    elif size_percent is not None:

        if isinstance(size_percent, (tuple, list, np.ndarray)):
            size_percent = np.array(size_percent).astype(np.float)
        else:
            size_percent = np.array([size_percent] * 3, dtype=np.float)

        if len(size_percent) != 3 or not (np.all(size_percent <= 1)
                                          and np.all(size_percent >= 0)):
            raise ValueError("Argument 'size_percent' must be array"
                             + " like of size 3 or float. "
                             + "Got {}.".format(size_percent))
        size = np.ceil(np.array(image.shape) * size_percent).astype(np.int)
    else:

        if isinstance(size, (tuple, list, np.ndarray)):
            size = np.array(size).astype(np.int)
        else:
            size = np.array([size] * 3, dtype=np.int)

        if len(size) != 3 or np.any(size <= 0):
            raise ValueError("Argument 'size' must be array"
                             + " like of size 3 or int value. Got {}.".format(size))


    return coarse_salt_and_pepper_numba(image, size, p, insert_value, coarse)
