import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

from .typing_details import Array

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

    cmin = masked_collection.min()
    cmax = masked_collection.max()

    return masked_collection, cmin, cmax


def save_or_show(save_path: str):
    """Saves matplotlib figure if save path is provided."""
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=SAVE_DPI)
        plt.clf()


def view_center_slices(image_volumes: Array[float, ('T','...')], mask_volume: Array[bool, ('...')], 
                       color_scale: bool = False, title_names: list[str] = None, 
                       full_save_path: str = None):
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

    elif abs(len(image_volumes.shape) - len(mask_volume.shape)) > 1:
        raise ValueError("Inputs 'image_volume' and 'mask_volume' differ by more than 2 axes")

    T, *dims = image_volumes.shape
    fig, axs = plt.subplots(T, 3)

    # promote axes
    axs = np.expand_dims(axs,0) if T == 1 else axs

    axes_slices = []
    cmin, cmax = np.inf, -np.inf
    for i in range(3):
        imgs_slice = np.take(image_volumes, dims[i]//2, i+1)
        mask_slice = np.take(mask_volume, dims[i]//2, i)

        imgs_mask_sl, cmin_i, cmax_i = apply_mask(imgs_slice, mask_slice)

        cmin = min(cmin, cmin_i)
        cmax = max(cmax, cmax_i)
        axes_slices.append(imgs_mask_sl)

    title_names = fill_list(title_names, 3*T)
    title_names = np.array(title_names).reshape(T, 3)

    if not color_scale:
        cmin, cmax = (None, None)

    for i in range(T):
        for j in range(3):
            ax = axs[i,j]
            tl_name = title_names[i,j]
            img_slice = axes_slices[j][i]

            ax.imshow(img_slice, vmin=cmin, vmax=cmax, interpolation='none')
            ax.axis('off')
            ax.set_title(tl_name)

    save_or_show(full_save_path)


def view_class_map(w_vol: Array[float, ('K','...')], slice_ind: int = None, slice_ax: int = 0,
                   order: Array[int, 'K'] = None, full_save_path: str = None):
    K, *spat_shp = w_vol.shape

    if slice_ind is None:
        slice_ind = spat_shp[slice_ax]//2

    if order is not None:  # for example if order = np.argsort(params.mu[seq_id])
        w_vol = w_vol[order]

    w_cls = np.argmax(np.take(w_vol, slice_ind, slice_ax+1), axis=0)
    cmap = plt.get_cmap('tab10', K)
    colors = cmap.colors

    plt.imshow(w_cls, cmap=cmap)
    plt.axis('off')
    plt.legend(handles=[Patch(facecolor=colors[i],label=rf'cls ${i}$') for i in range(K)])

    save_or_show(full_save_path)


def view_distributions(image_volumes: Array[float, ('T','...')], bandwidth: float,
                       full_save_path: str = None):
    image_volumes = np.asarray(image_volumes)
    xmin = image_volumes.min()
    xmax = image_volumes.max()

    x = np.linspace(xmin, xmax, 200)

    for image in image_volumes:
        img_kde = gaussian_kde(image.flatten(), bandwidth)
        plt.plot(x, img_kde(x))

    save_or_show(full_save_path)


def fill_list(vals: Any, length:int):
    try:
        vals[0]
    except TypeError:
        vals = [vals]

    repeat_mult = length//len(vals)
    vals = vals * repeat_mult

    len_rem = length - len(vals)
    vals = vals + vals[:len_rem]
    
    return vals
