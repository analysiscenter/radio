import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

def get_pixel_coords(nodules):
    """ Get nodules info in pixel coords from nodules recarray.
    """
    coords = (nodules.nodule_center - nodules.origin) / nodules.spacing
    diams = np.ceil(nodules.nodule_size / nodules.spacing)
    nodules = np.rint(np.hstack([coords, diams])).astype(np.int)
    return nodules

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

def show_images(mnist_batch):
    """ Show images from ImageBatch with mnist
    """
    imgs = mnist_batch.images[..., 0]
    _, axes = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        axes[i].imshow(img, cmap=plt.cm.gray)

def load_example(path, radio, fmt='blosc', ix='1.3.6.1.4.1.14519.5.2.1.6279.6001.621916089407825046337959219998'):
    """ Load example batch from disk
    """
    from radio import CTImagesMaskedBatch as CTIMB
    from radio.batchflow import Pipeline, FilesIndex, Dataset
    path = os.path.join(path, ix)
    luna_index = FilesIndex(path=path, dirs=True)
    lunaset =  Dataset(luna_index, batch_class=CTIMB)
    load_ppl = (Pipeline()
                 .load(fmt='blosc', components=['images', 'spacing', 'origin', 'masks']) << lunaset)

    btch = load_ppl.next_batch(1)
    return btch

def get_nodules_pixel_coords(batch):
    """ get numpy array of nodules-locations and diameter in relative coords
    """
    nodules_dict = dict()
    nodules_dict.update(numeric_ix=batch.nodules.patient_pos)
    pixel_zyx = np.rint((batch.nodules.nodule_center - batch.nodules.origin) / batch.nodules.spacing).astype(np.int)
    nodules_dict.update({'coord' + letter: pixel_zyx[:, i] for i, letter in enumerate(['Z', 'Y', 'X'])})
    nodules_dict.update({'diameter_pixels': (np.rint(batch.nodules.nodule_size / batch.nodules.spacing).mean(axis=1)
                                             .astype(np.int))})
    pixel_nodules_df = pd.DataFrame.from_dict(nodules_dict).loc[:, ('numeric_ix', 'coordZ', 'coordY',
                                                                    'coordX', 'diameter_pixels')]
    return pixel_nodules_df

def num_of_cancerous_pixels(batch, max_num=10):
    """ Calculate number of cancerous pixels in items from batch
    """
    stats = dict()
    n_print = min(max_num, len(batch))
    for i in range(n_print):
        stats.update({'Scan ' + str(i): int(np.sum(batch.get(i, 'masks')))})

    stats = {'Number of cancerous  pixels: ': stats}
    stats_df = pd.DataFrame.from_dict(stats, orient='index').loc[:, ['Scan '+ str(i) for i in range(n_print)]]
    return stats_df

def show_slices(batches, scan_indices, ns_slice, grid=True, **kwargs):
    """ Plot slice with number n_slice from scan with index given by scan_index from batch
    """
    font_caption = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'normal',
                    'size': 18}
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 15}

    # fetch some arguments, make iterables out of args
    def iterize(arg):
        return arg if isinstance(arg, (list, tuple)) else (arg, )

    components = kwargs.get('components', 'images')
    batches, scan_indices, ns_slice, components  = [iterize(arg) for arg in (batches, scan_indices,
                                                                             ns_slice, components)]
    clims = kwargs.get('clims', (-1200, 300))
    clims = clims if isinstance(clims[0], (tuple, list)) else (clims, )

    # lengthen args
    n_boxes = max(len(arg) for arg in (batches, scan_indices, ns_slice, clims))
    def lengthen(arg):
        return arg if len(arg) == n_boxes else arg * n_boxes

    batches, scan_indices, ns_slice, clims, components = [lengthen(arg) for arg in (batches, scan_indices, ns_slice,
                                                                                    clims, components)]

    # plot slices
    _, axes = plt.subplots(1, n_boxes, squeeze=False, figsize=(10, 4 * n_boxes))

    zipped = zip(range(n_boxes), batches, scan_indices, ns_slice, clims, components)

    for i, batch, scan_index, n_slice, clim, component in zipped:
        slc = batch.get(scan_index, component)[n_slice]
        axes[0][i].imshow(slc, cmap=plt.cm.gray, clim=clim)
        axes[0][i].set_xlabel('Shape: {}'.format(slc.shape[1]), fontdict=font)
        axes[0][i].set_ylabel('Shape: {}'.format(slc.shape[0]), fontdict=font)
        title = 'Scan' if component == 'images' else 'Mask'
        axes[0][i].set_title('{} #{}, slice #{} \n \n'.format(title, scan_index, n_slice), fontdict=font_caption)
        axes[0][i].text(0.2, -0.25, 'Total slices: {}'.format(len(batch.get(scan_index, component))),
                        fontdict=font_caption, transform=axes[0][i].transAxes)

        # set inverse-spacing grid
        if grid:
            inv_spacing = 1 / batch.get(scan_index, 'spacing').reshape(-1)[1:]
            step_mult = 50
            xticks = np.arange(0, slc.shape[0], step_mult * inv_spacing[0])
            yticks = np.arange(0, slc.shape[1], step_mult * inv_spacing[1])
            axes[0][i].set_xticks(xticks, minor=True)
            axes[0][i].set_yticks(yticks, minor=True)
            axes[0][i].set_xticks([], minor=False)
            axes[0][i].set_yticks([], minor=False)

            axes[0][i].grid(color='r', linewidth=1.5, alpha=0.5, which='minor')


    plt.show()
