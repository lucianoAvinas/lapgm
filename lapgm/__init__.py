import numpy as np
from typing import Callable
from types import ModuleType

# for __init__ package use
from lapgm import config
from lapgm import bias_calc
from lapgm import param_setup
from lapgm import lapgm_estim
from lapgm import laplacian_routines

# function shortcuts
from .typing_details import Array
from .param_setup import ParameterEstimate
from .lapgm_estim import LapGM, check_seq_shape
from .view_utils import view_center_slices, view_class_map, view_distributions


def module_set(mod: ModuleType, attr_nms: tuple[str], options: dict[str, ModuleType]):
    """Helper function for quick attribute assignment on submodules"""
    for attr_nm in attr_nms:
        setattr(mod, attr_nm, options[attr_nm])


def use_gpu(assign_bool: bool):
    """Propagate GPU or CPU settings to lapgm submodules"""
    if assign_bool:
        if config._CUPY_PATH is None:
            raise ModuleNotFoundError('CuPy array package not found')

        # cupy related packages
        import cupy as ap
        import cupyx.scipy.sparse as sp
        import cupyx.scipy.sparse.linalg as spl
        from cupyx.scipy.ndimage import zoom as zoom
        from .cupyx_mvn import multivariate_normal as multi_normal

    else:
        # numpy related packages
        import numpy as ap
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        from scipy.ndimage import zoom as zoom
        from scipy.stats import multivariate_normal as multi_normal

    options = dict(ap=ap, sp=sp, spl=spl, zoom=zoom, multi_normal=multi_normal)

    # dynamically set import for submodules
    module_set(bias_calc, ('ap', 'sp', 'multi_normal'), options)
    module_set(param_setup, ('ap', 'spl'), options)
    module_set(lapgm_estim, ('ap', 'zoom'), options)
    module_set(laplacian_routines, ('ap', 'sp'), options)


### Initialize preferred array package to Numpy CPU ###
use_gpu(False)


def debias(image: Array[float, ('M', '...')], params: ParameterEstimate):
    nonzero_mask = (image > 0)
    deb_image = np.zeros(image.shape)

    image_nz = image[nonzero_mask]
    B_nz = params.B[nonzero_mask]

    deb_image[nonzero_mask] = np.exp(np.log(image_nz) - B_nz)
    return deb_image


def normalize(image: Array[float, ('M', '...')], n_seqs: int, 
              params: ParameterEstimate, target_intensity: float = 1000., 
              norm_fn: Callable = None,  per_seq_norm: bool = True):

    image = check_seq_shape(image, n_seqs)

    if norm_fn is None:
        norm_fn = max_norm_fn
    
    if per_seq_norm:
        norm_image = np.zeros(image.shape)

        for i in range(n_seqs):
            sc = norm_fn(image[i], params, i)
            norm_image[i] = image[i] / sc * target_intensity
    else:
        sc = norm_fn(image, params)
        norm_image = image / sc  * target_intensity

    return norm_image


def max_norm_fn(image: Array[float, ('M', '...')], params: ParameterEstimate, 
                seq_id: int = None):

    mu = params.mu
    if seq_id is None:
        mu_mx = np.exp(np.max(mu, axis=1))

        # Least-squares estimate on [TARGT,...,TARGT]
        sc = (mu_mx @ mu_mx) / np.sum(mu_mx)
    else:
        sc = np.exp(np.max(mu[:,seq_id]))

    return sc
