from typing import Callable

# CPU specification on default
import numpy as ap
from scipy.ndimage import zoom
_USE_GPU = False

from lapgm.typing_utils import Array
from lapgm.bias_calc import compute_bias_field
from lapgm.param_setup import ParameterEstimate
from lapgm.laplacian_routines import prepare_wgts, hyper_ellipsoid_radius, \
                                     construct_dirichlet_lap, weight_laplacian


def set_compute(assign_bool: bool):
    """Set preferred array package"""
    global _USE_GPU, ap, zoom
    
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
        self.weight_settings = None

    def specify_cylindrical_decay(self, alpha, axes_of_symmetry: tuple[int] = (0,), 
                                  center: tuple[int] = (0,0), semi_axes: tuple[int] = (1,1)):
        wgt_func = lambda x: x**(-alpha)
        self.weight_settings = dict(wgt_func=wgt_func, axes_of_symmetry=axes_of_symmetry, 
                                    center=center, semi_axes=semi_axes)

    def set_hyperparameters(self, n_classes: int, tau: float, log_initialize: bool = True, 
                            kmeans_n_init: int = 5, kmeans_max_iters: int = 300):
        self.tau = tau
        self.init_settings = dict(n_classes=n_classes, log_initialize=log_initialize, 
                                  n_init=kmeans_n_init, max_iters=kmeans_max_iters)

    def estimate_parameters(self, image: Array[float, ('M', '...')], n_seqs: int = 1, 
                            bias_tol: float = 1e-4, max_em_iters: int = 25, 
                            max_cg_iters: int = 25000, random_seed: int = 1,  
                            print_tols: bool = False, params_obj: ParameterEstimate = None, 
                            custom_wgts: Array[float, 'N'] = None):

        # set max_em_iters to zero to get just inital_param estimate and then modify
        # mention this in documentation of params_obj

        # check if number of sequences matches first axis length
        image = check_seq_shape(image, n_seqs)

        # cast to relevant array
        image = ap.asarray(image)        

        # dimensions of the spatial axes
        spat_shape = image.shape[1:]

        # downscale image
        per_seq_sc = [1] + [self.scale_fctr]*len(spat_shape)
        image = abs(zoom(image, per_seq_sc, order=self.scaling_smoothness, mode='nearest'))        

        # flatten spatial axes and log
        spat_ds_shape = image.shape[1:]
        log_image = fill_zeros_and_log(image).reshape(n_seqs, -1)

        # construct initial parameter estimate
        params = ParameterEstimate(params_obj, image, log_image, self.init_settings, 
                                   dict(maxiter=max_cg_iters), self.save_settings)

        # apply laplacian weighting routine
        if custom_wgts is not None:
            wgts = ap.asarray(custom_wgts)

        elif self.weight_settings is not None:
            coord_transf = hyper_ellipsoid_radius(self.weight_settings['center'], 
                                                  self.weight_settings['semi_axes'])
            wgts =  prepare_wgts(spat_ds_shape, self.weight_settings['wgt_func'], 
                                 coord_transf, self.weight_settings['axes_of_symmetry'])
        else:
            # uniform multiplicative weight if no weight is provided
            wgts = ap.ones(log_image.shape[1])

        # construct weighted Laplacian and scale by tau
        L = weight_laplacian(construct_dirichlet_lap(spat_ds_shape, True), wgts) / self.tau

        # calculate parameter estimates
        params = compute_bias_field(log_image, L, params, bias_tol, max_em_iters, 
                                    random_seed, print_tols)
        
        # define upscaler and apply
        upscaler = lambda img_ds, targ_shp, sc_lst: scale_and_pad(img_ds, targ_shp, sc_lst, 
                                                                  self.scaling_smoothness)
        params.upscale_parameters(upscaler, spat_shape, 1/self.scale_fctr)

        return params


def check_seq_shape(image: Array[Any, ('M', '...')], n_seqs: int):
    if image.shape[0] != n_seqs:
        if n_seqs == 1:
            image = image[None]
        else:
            raise ValueError(f'User specifed {n_seqs} sequences but image has dimension '
                             f'{image.shape[0]} on the first axis.')
    return image


def fill_zeros_and_log(image: Array[Any, ('...')]):
    zero_mask = (image == 0)

    min_val = ap.min(image[~zero_mask])//2
    zero_cnt = ap.sum(zero_mask).item()
    image[zero_mask] = ap.random.random(zero_cnt) * min_val

    return ap.log(image)


def scale_and_pad(scale_img: Array[float, ('...')], orig_dims: tuple[int], 
                  scale_fctrs: tuple[float], smooth_order: int):
    """
        scale_img: downscaled image
        orig_dim: dimensions of original image
        scale_fctrs: tuple of scaling factors for per-axis downscale/upscale
        smooth_order: order of smoothness to interpolate with
    """
        
    img = abs(zoom(scale_img, scale_fctrs, order=smooth_order, mode='nearest'))

    # calculate dimension offset between upsampled image and original image
    dim_diff = tuple(o-n for o,n in zip(orig_dims, img.shape))

    # pad smaller dims
    img = ap.pad(img, tuple((0, max(dim, 0)) for dim in dim_diff), 'edge')

    # crop bigger dims
    img = img[tuple([slice(0, dim if dim < 0 else None) for dim in dim_diff])]

    return img
