from typing import Callable

# CPU specification on default
import numpy as ap
from scipy.ndimage import zoom
_USE_GPU = False

from lapgm.typing_utils import Array
from lapgm.param_setup import ParameterEstimate
from lapgm.laplacian_routines import prepare_wgts, hyper_ellipsoid_radius


def set_compute(assign_bool: bool):
    """Set preferred array package"""
    global _USE_GPU, ap, zppm
    
    if assign_bool:
        import cupy as ap
        from cupyx.scipy.ndimage import zoom
        _USE_GPU = True

    else:
        import numpy as ap
        from scipy.ndimage import zoom
        _USE_GPU = False


class LapGM:
    def __init__(self, downscale_factor: int = 1, scaling_order: int = 3, 
                 store_history: bool = False, unstore_large: bool = True):
        self.scale_fctr = 1/downscale_factor
        self.scaling_smoothness = scaling_order
        self.save_settings = dict(store_history=store_history, unstore_large=unstore_large)

        self.tau = None
        self.init_settings = None
        self.laplacian_weighting_fn = None

    def specify_laplacian_weighting(self, wgt_func: Callable, axes_of_symmetry: tuple[int] = None, 
                                    center: tuple[int] = None, semi_axes: tuple[int] = None):
        # center and semi_axes are done using the reduced dimensions

        coord_transf = hyper_ellipsoid_radius(centers, semi_axes)

        self.laplacian_weighting_fn = lambda shp: prepare_wgts(shp, wgt_func, coord_transf, 
                                                               axes_of_symmetry)

    def set_hyperparameters(n_classes: int, tau: float, log_initialize: bool = True, kmeans_n_init: int = 10,
                            kmeans_max_iters: int = 300):
        self.n_classes = n_classes
        self.tau = tau
        self.init_settings = dict(log_initialize=log_initialize, n_init=kmeans_n_init, 
                                  max_iters=kmeans_max_iters)

    def estimate_parameters(self, image: Array[float, ('M', '...')], n_seqs: int, bias_tol: float = 1e-4,
                            max_em_iters: int = 25, max_cg_iters: int = 25000, random_seed: int = 1, 
                            print_tols: bool = False, params_obj: type[ParameterEstimate] = None, 
                            custom_wgts: Array[float, 'N'] = None):
        # try and catch the usual overflow error
        # set max_em_iters to zero to get just inital_param estimate and then modify

        # cast to relevant array
        image = ap.asarray(image)
        image = check_seq_shape(image, n_seqs)

        # dimensions of the spatial axes
        spat_shape = image.shape[1:]

        # downscale image
        per_seq_sc = [1] + [self.scale_fctr]*len(spat_shape)
        image = zoom(image, per_seq_sc, order=self.scaling_smoothness, mode='nearest')

        # flatten spatial axes
        spat_ds_shape = image.shape[1:]
        image = image.reshape(n_seqs, -1)
        log_image = fill_zeros_and_log(image)

        init_params = dict() if init_params is None else init_params
        params = ParameterEstimate(params_obj, image, log_image, self.init_settings, 
                                   dict(maxiter=max_cg_iters), self.save_settings)

        if custom_wgts is not None:
            wgts = ap.asarray(custom_wgts)
        else:
            wgts = self.laplacian_weighting_fn(spat_ds_shape)

        # weight Laplacian and scale by tau
        L = weight_laplacian(construct_dirichlet_lap(spat_ds_shape, True), wgts) / self.tau

        # estimate parameters
        params = compute_bias_field(log_image, L, params, bias_tol, max_em_iters, random_seed, 
                                    print_tols)
        
        # define upscaler
        upscaler = lambda img_ds, scale_fctrs: scale_and_pad(img_ds, spat_shape, scale_fctrs, 
                                                             self.scaling_smoothness)

        
        params.apply_upscaler_and_offload(upscaler, spat_ds_shape, self.scale_fctr)

        return params


def check_seq_shape(image, n_seqs):
    if image.shape[0] != n_seqs:
        if n_seqs == 1:
            image = image[None]
        else:
            raise ValueError(f'User specifed {n_seqs} sequences but image has dimension '
                             f'{image.shape[0]} on the first axis.')
    return image


def fill_zeros_and_log(image):
    zero_mask = (image == 0)
    nonzero_count = reduce(mul, image.shape) - ap.sum(zero_mask)

    min_val = ap.min(image[~zero_mask])
    image[zero_mask] = ap.random.random(nonzero_count) * min_val

    return ap.log(image)


def scale_and_pad(scale_img: Array[float, ('...')], orig_dims: tuple[int], 
                  scale_fctrs: tuple[float], smooth_order: int):
    """
        scale_img: downscaled image
        orig_dim: dimensions of original image
        scale_fctrs: tuple of scaling factors for per-axis downscale/upscale
        smooth_order: order of smoothness to interpolate with
    """
        
    img = zoom(scale_img, scale_fctrs, order=smooth_order, mode='nearest')

    # calculate dimension offset between upsampled image and original image
    dim_diff = tuple(o-n for o,n in zip(orig_dims, img.shape))

    # pad smaller dims
    img = ap.pad(img, tuple((0, max(dim, 0)) for dim in dim_diff), 'edge')

    # crop bigger dims
    img = img[tuple([slice(0, dim if dim < 0 else None) for dim in dim_diff])]

    return img
