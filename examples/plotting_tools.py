import numpy as np
from PIL import Image

def trim_cast_uint8(array, lim=None):
    # trim
    lim = lim if lim is not None else (np.min(array), np.max(array))
    array = np.where(array <= lim[1], array, lim[1])
    array = np.where(array >= lim[0], array, lim[0])

    # cast
    array = np.rint((array - lim[0]) / (lim[1] - lim[0]) * 255).astype(np.uint8)
    return array


def pil_plot_slices(height, *arrays, lims=None):
    lims = lims if lims is not None else (None, ) * len(arrays)
    data = []
    for a, lim in zip(arrays, lims):
        n_slice = int(a.shape[0] * height)
        data.append(trim_cast_uint8(a[n_slice], lim))

    data = np.concatenate(data, axis=1)
    return Image.fromarray(data)


def combine_in_rgb(masks, supress=(False, False, False)):
    """ Masks: iterable of three 2d-masks.
    """
    colors = list(np.identity(3, dtype=np.uint8) * 255)
    blended = np.zeros(shape=masks[0].shape + (3, ), dtype=np.uint8)
    for mask, color, sup in zip(masks, colors, supress):
        if not sup:
            img = Image.fromarray(mask, 'L').convert('RGB')
            rgba = np.array(img)
            white_mask = (rgba[..., 0] == 255) & (rgba[..., 1] == 255) & (rgba[..., 2] == 255)
            blended[..., :][white_mask] = tuple(color)

    return blended

def blend_mask_to_scan(scan, mask, alpha=0.5):
    scan_masked = scan.copy().astype(np.int32)
    scan_masked += mask
    scan_masked = trim_cast_uint8(scan_masked, (0, 255))
    return Image.blend(Image.fromarray(scan, 'RGB'), Image.fromarray(scan_masked, 'RGB'), alpha)
