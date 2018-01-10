import matplotlib.pyplot as plt
import numpy as np

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

    batches, scan_indices, ns_slice  = [iterize(arg) for arg in (batches, scan_indices, ns_slice)]
    clims = kwargs.get('clims', (-1200, 300))
    clims = clims if isinstance(clims[0], (tuple, list)) else (clims, )

    # lengthen args
    n_boxes = max(len(arg) for arg in (batches, scan_indices, ns_slice, clims))
    def lengthen(arg):
        return arg if len(arg) == n_boxes else arg * n_boxes

    batches, scan_indices, ns_slice, clims = [lengthen(arg) for arg in (batches, scan_indices, ns_slice, clims)]

    # plot slices
    _, axes = plt.subplots(1, n_boxes, squeeze=False, figsize=(10, 4 * n_boxes))

    for i, batch, scan_index, n_slice, clim in zip(range(n_boxes), batches, scan_indices, ns_slice, clims):
        slc = batch.get(scan_index, 'images')[n_slice]
        axes[0][i].imshow(slc, cmap=plt.cm.gray, clim=clim)
        axes[0][i].set_xlabel('Shape: {}'.format(slc.shape[1]), fontdict=font)
        axes[0][i].set_ylabel('Shape: {}'.format(slc.shape[0]), fontdict=font)
        axes[0][i].set_title('Scan #{}, slice #{} \n \n'.format(i, n_slice), fontdict=font_caption)
        axes[0][i].text(0.2, -0.25, 'Total slices: {}'.format(len(batch.get(scan_index, 'images'))),
                        fontdict=font_caption, transform=axes[0][i].transAxes)

        # set inverse-spacing grid
        if grid:
            inv_spacing = 1 / batch.get(scan_index, 'spacing').reshape(-1)[1:]
            print(inv_spacing)
            step_mult = 50
            xticks = np.arange(0, slc.shape[0], step_mult * inv_spacing[0])
            yticks = np.arange(0, slc.shape[1], step_mult * inv_spacing[1])
            axes[0][i].set_xticks(xticks, minor=True)
            axes[0][i].set_yticks(yticks, minor=True)
            axes[0][i].set_xticks([], minor=False)
            axes[0][i].set_yticks([], minor=False)

            axes[0][i].grid(color='r', linewidth=1.5, alpha=0.5, which='minor')


    plt.show()
