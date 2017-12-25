import matplotlib.pyplot as plt

def show_slice(batch, scan_index, n_slice, **kwargs):
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

    slc = batch.get(scan_index, 'images')[n_slice]
    _, axes = plt.subplots(1, 1, squeeze=False, figsize=(10, 4))
    axes[0][0].imshow(slc, cmap=plt.cm.gray, clim=kwargs.get('clim', (-1200, 300)))
    axes[0][0].set_xlabel('Shape: {}'.format(slc.shape[1]), fontdict=font)
    axes[0][0].set_ylabel('Shape: {}'.format(slc.shape[0]), fontdict=font)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    axes[0][0].set_title('Slice #{} \n \n'.format(n_slice), fontdict=font_caption)
    axes[0][0].text(0.15, -0.25, 'Total slices: {}'.format(len(batch.get(scan_index, 'images'))),
                    fontdict=font_caption,
                    verticalalignment='center', transform=axes[0][0].transAxes)

    plt.show()
