from typing import Callable

# CPU specification on default
import numpy as ap
from scipy.ndimage import zoom
_USE_GPU = False

from lapgm.typing_utils import Array
from lapgm.param_utils import ParameterEstimate
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



## Upscale downscale routine should happen 

## Takes a dict of init param. If no init carry on with usual initialization

## User should not have to handle lap

## Provide a set weighting scheme. If no weighting do default

## Usage:
## lpg = LapGM(args)   ## Meta arguments: Like save history, overwrite conditions, clear and offload conditions
#                                         can also determine if image array should be saved or ignored.
#                                         offload option affects both history and returned estimate
## lpg.set_laplacian_wgts(...)  ## Good -> this should actually defer something for ParamEstimate 
#                               # check a self variable at .run which then generates lap matrix in ParamEstimate.
## **kwargs -> manual init, for clarity this should happen in run. Logic to then determine what remaining should be init'd
### Actually let init_params just be a dict which updates over or cancels init call (if params are correctly accounted for)

## lpgm.run(image, (run options like, downsample seed,iterations) )  [[Look into saving, separate duties]]
#### return ParameterEstimate, handle down/up sampling in run. This would imply .run directly modifying ParameterEstimate
#### ParameterEstimate should have .estimate, .history attributes. These are self scope and initialize to None


#### Stil pass in Parameter estimate, create internally in LapGM. Takes care of saving 
#### upscale all history images at end of run. This gives LapGM upscaling responsibility.

class LapGM:
    def __init__(self, downscale_factor: int = 1, scaling_order: int = 3, 
                 store_history: bool = False, unstore_large: bool = True):
        self.scale_fctr = 1/downscale_factor
        self.scaling_smoothness = scaling_order
        self.store_history = store_history
        self.unstore_large = unstore_large

        self.tau = None
        self.initializer_kwargs = None
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
        self.initializer_kwargs = dict(log_initialize=log_initialize, n_init=kmeans_n_init, 
                                       max_iters=kmeans_max_iters)

    def estimate_params(self, image: Array[float, ('M', '...')], n_seqs: int, bias_tol: float = 1e-4,
                        max_em_iters: int = 25, max_cg_iters: int = 25000, random_seed: int = 1, 
                        print_tols: bool = False, init_params: type[ParameterEstimate] = None, 
                        custom_wgts: Array[float, 'N'] = None):
        # try and catch the usual overflow error
        # only cast back to cpu for kmeans calc
        # log scaling init and the eventual 0 pass the comes with it # let init settings do this
        # casting will happen here as scale and pad needs appropriate dim anywas
        # create parameter object here
        # params holds laplacian that is augmented by tau
        # flatten, gpu cast, and reshape here
        # set max_em_iters to zero to get just inital_param estimate and then modify

        # first create L_upper, if custom_wgts then done make weighted lap, else check
        # weighting fn, if none do ap.ones
        # if _USE_GPU:
        # wgts = ap.asarray(wgts) for the custom wgts; maybe always just use .asarray regardless of GPU


        # cast to relevant array
        image = ap.asarray(image)

        if image.shape[0] != n_seqs:
            if n_seqs == 1:
                image = image[None]
            else:
                raise ValueError(f'User specifed {n_seqs} sequences but image has dimension '
                                 f'{image.shape[0]} on the first axis.')

        # dimensions of the spatial axes
        spat_shape = image.shape[1:]

        # downscale image
        per_seq_sc = [1] + [self.scale_fctr]*(len(spat_shape)-1)
        image = zoom(image, per_seq_sc, order=smooth_order, mode='nearest')

        # flatten spatial axes
        spat_ds_shape = image.shape[1:]
        image = image.reshape(n_seqs, -1)

        init_params = dict() if init_params is None else init_params
        params = ParameterEstimate(image, init_params, self.log_initialize, self.max_cg_iters
                                   self.store_history, self.unstore_large)



        



# params.offload_gpu()

# Reshape to appropriate spatial dimensions
# params.B = params.B.reshape(params.img_shape)
# params.w = params.w.reshape(params.n_classes, *params.img_shape)

"""
def offload_gpu(self):
        if _USE_GPU:
            self.pi = cp.asnumpy(self.pi)
            self.mu = cp.asnumpy(self.mu)
            self.Sigma = cp.asnumpy(self.Sigma)
            self.w = cp.asnumpy(self.w)
            self.B = cp.asnumpy(self.B)
"""

# for loop on curr params and for loop on history

# 
        # include debias utility function and normalization utility function here
        # will just take a parameter estimate and image to run


def scale_and_pad(scale_img: Array[float, ('...')], orig_dims: Iterable[int], 
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
