import numpy as np
from PIL import Image

def trim_cast_uint8(array, lim=None):
    """ Trim an array using lim as limits, transform its range to [0, 255] and
    cast the array to uint8.
    """
    # trim
    lim = lim if lim is not None else (np.min(array), np.max(array))
    array = np.where(array <= lim[1], array, lim[1])
    array = np.where(array >= lim[0], array, lim[0])

    # cast
    array = np.rint((array - lim[0]) / (lim[1] - lim[0]) * 255).astype(np.uint8)
    return array


def pil_plot_slices(height, *arrays, lims=None):
    """ Plot slices of several 3d-np.arrays using PIL.
    """
    lims = lims if lims is not None else (None, ) * len(arrays)
    data = []
    for a, lim in zip(arrays, lims):
        n_slice = int(a.shape[0] * height)
        data.append(trim_cast_uint8(a[n_slice], lim))

    data = np.concatenate(data, axis=1)
    return Image.fromarray(data)


def combine_in_rgb(masks, supress=(False, False, False)):
    """ Combine in rgb three 2d-masks. Supress any of them, if needed.
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
    """ Blend a 2d rgb-mask with 2d-scan.
    """
    scan_masked = scan.copy().astype(np.int32)
    scan_masked += mask
    scan_masked = trim_cast_uint8(scan_masked, (0, 255))
    return Image.blend(Image.fromarray(scan, 'RGB'), Image.fromarray(scan_masked, 'RGB'), alpha)

def apply_masks(scan_3d, masks_3d, height, supress=(False, False, False), alpha=0.5,
                shape=(384, 384)):
    """ Put 3d-mask on 3d-scan. Resize to given shape if needed.
    Auxilliary function for convenient usage of interact.
    """
    depth = scan_3d.shape[0]
    n_slice = int(depth * height)

    # pick slices
    masks = [m[n_slice] if m is not None else None for m in masks_3d]
    scan = scan_3d[n_slice]

    sup = []
    for s, m in zip(supress, masks):
        sup.append(s or (m is None))

    combined = combine_in_rgb(masks, sup)

    scan_rgb = np.array(Image.fromarray(scan, 'L').convert('RGB'))
    img = blend_mask_to_scan(scan_rgb, combined, alpha)
    if shape is not None:
        return img.resize(shape, Image.BILINEAR)
    else:
        return img
