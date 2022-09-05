import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from scipy.stats import gaussian_kde
from lapgm.typing_details import Array

SAVE_DPI = 300


def apply_mask(image_collection: Array[float, ('...', 'X')], mask: Array[bool, 'X'] = None):
    """
        image_collection: arbitrary size array of float or complex data type
        mask: boolean array with same tail axes size as image_collection.
              Is used to compute maximum and minimum values of image_collection
              within the mask. For complex data the modulus is used for min/max values.
    """

    if mask is None:
        masked_collection = image_collection
    else:
        masked_values = image_collection[...,mask]

        masked_collection = np.zeros_like(image_collection)
        masked_collection[...,mask] = masked_values

    return masked_collection


def save_or_show(save_path: str):
    """Saves matplotlib figure if save path is provided."""
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=SAVE_DPI)
        plt.close()
        plt.clf()


def view_center_slices(image_volumes: Array[float, ('T','...')], mask_volume: Array[bool, ('...')] = None, 
                       title_names: list[str] = None, cmap_name: str = 'viridis', full_save_path: str = None):
    """
        image_volumes: collection of 3 axes arrays
        mask: boolean array with same spatial size as image volume. Emphasizes important regions
        title_names: list of subplot title names
        full_save_path: full path and file name to save to
    """

    image_volumes = np.asarray(image_volumes)

    if len(image_volumes.shape) == 3:
        image_volumes = np.expand_dims(image_volumes, 0)

    elif len(image_volumes.shape) < 3:
        raise ValueError(f"Input 'image_volume' should have at least 3 axes")

    T, *dims = image_volumes.shape
    image_maskd = apply_mask(image_volumes, mask_volume)

    axes_slices = []
    for i in range(3):
        imgs_mask_sl = np.take(image_maskd, dims[i]//2, i+1)  
        axes_slices.append(imgs_mask_sl)

    fig = plt.figure()
    subfigs = fig.subfigures(nrows=T, ncols=1)
    title_names = fill_list(title_names, T)

    # Add dummy axis for single volume
    if T == 1:
        subfigs = [subfigs]

    for i, subfg in enumerate(subfigs):
        axs = subfg.subplots(nrows=1, ncols=3)
        for j, ax in enumerate(axs):
            img_slice = axes_slices[j][i]
            ax.imshow(np.flipud(img_slice), interpolation='none', cmap=cmap_name)
            ax.axis('off')

            if j == 1:
                ax.set_title(title_names[i])

    save_or_show(full_save_path)


def view_class_map(w_vol: Array[float, ('K','...')], order: Array[int, 'K'] = None, 
                   title_name: str = None, full_save_path: str = None):
    K, *spat_shp = w_vol.shape

    if order is not None: 
        w_vol = w_vol[order]

    w_vol = np.argmax(w_vol, axis=0)
    cmap = plt.get_cmap('tab10', K)

    d = len(spat_shp)
    fig = plt.figure()

    # Subfigure routine centers the image triplet
    subfig = fig.subfigures(nrows=1, ncols=1)
    axs = subfig.subplots(nrows=1, ncols=d)

    for i in range(d):
        ax = axs[i]
        ax.imshow(np.flipud(np.take(w_vol, spat_shp[i]//2, i)), cmap=cmap, interpolation='none')
        ax.axis('off')

        if i == 1:
            ax.set_title(title_name)

    save_or_show(full_save_path)


def view_distributions(image_volumes: Array[float, ('T','...')], bandwidth: float,
                       mask_volume: Array[bool, ('...')] = None, title_name: str = None, 
                       combine=False, full_save_path: str = None):
    image_volumes = np.asarray(image_volumes)

    if len(image_volumes.shape) == 3:
        image_volumes = np.expand_dims(image_volumes, 0)

    elif len(image_volumes.shape) < 3:
        raise ValueError(f"Input 'image_volume' should have at least 3 axes")

    image_volumes = image_volumes[...,mask_volume]

    xmin = image_volumes.min()
    xmax = image_volumes.max()

    x = np.linspace(xmin, xmax, 250)

    if combine:
        image_volumes = image_volumes.flatten()[None]

    for image in image_volumes:
        img_kde = gaussian_kde(image, bandwidth)
        plt.plot(x, img_kde(x))

    plt.title(title_name)

    save_or_show(full_save_path)


def fill_list(vals: Any, length:int):
    try:
        vals[0]
    except TypeError:
        vals = [vals]*length
    
    return vals
