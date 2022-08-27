import sys
import warnings
import numpy as np
from typing import Callable
from types import ModuleType

# function shortcuts
from .typing_details import Array
from .param_setup import ParameterEstimate
from .lapgm_estim import LapGM, check_seq_shape
from .view_utils import view_center_slices, view_class_map, view_distributions

# for __init__ array package assignment
from lapgm import bias_calc
from lapgm import param_setup
from lapgm import lapgm_estim
from lapgm import laplacian_routines

## CPU array packages
import numpy as ap_cpu
import scipy.sparse as sp_cpu
import scipy.sparse.linalg as spl_cpu
from scipy.ndimage import zoom as zoom_cpu
from scipy.stats import multivariate_normal as multi_normal_cpu

## GPU array packages
try:
    import cupy as ap_gpu
    import cupyx.scipy.sparse as sp_gpu
    import cupyx.scipy.sparse.linalg as spl_gpu
    from cupyx.scipy.ndimage import zoom as zoom_gpu
    from .cupyx_mvn import multivariate_normal as multi_normal_gpu
    _USE_GPU = True

except ModuleNotFoundError as err:
    warnings.warn(f'Certain GPU array packages could not be found. ' + \
                   'lapgm.use_gpu will be restricted to cpu arrays only.')
    ERR = err
    _USE_GPU = False

# Naming conventions for CPU/GPU array packages
ARRAY_PKGS = ('ap', 'sp', 'spl', 'zoom', 'multi_normal')


def module_set(mod: ModuleType, attr_nms: tuple[str], options: dict[str, ModuleType]):
    """Helper function for quick attribute assignment on submodules."""
    for attr_nm in attr_nms:
        setattr(mod, attr_nm, options[attr_nm])


def use_gpu(assign_bool: bool):
    """Propagate GPU or CPU settings to lapgm submodules.

    Raises ModuleNotFoundError if user sets compute to GPU while missing relevant packages.
    """
    # map boolean value to appropriate pkg suffix
    if assign_bool:
        if _USE_GPU:
            pkg_suff = 'gpu'
        else:
            # catch invalid compute assignment
            raise ModuleNotFoundError(ERR)
    else:
        pkg_suff = 'cpu'

    # get module object
    module_self = sys.modules[__name__]

    # link array package names to cpu/gpu packages
    options = {pkg_nm:getattr(module_self, f'{pkg_nm}_{pkg_suff}') for pkg_nm in ARRAY_PKGS}

    # dynamically set import for submodules
    module_set(bias_calc, ('ap', 'sp', 'multi_normal'), options)
    module_set(param_setup, ('ap', 'spl'), options)
    module_set(lapgm_estim, ('ap', 'zoom'), options)
    module_set(laplacian_routines, ('ap', 'sp'), options)

    # additionally update compute metadata for param_setup
    param_setup._USE_GPU = _USE_GPU


### Initialize preferred array package to CPU ###
use_gpu(False)


def debias(image: Array[float, ('M', '...')], params: ParameterEstimate):
    """Debias image using previously computed LapGM parameters.

    Only non-negative intensity values will be considered for debiasing.

    Args:
        image: 'M' channel image with variable spatial dimensions.
        params: Previously computed LapGM parameters.

    Returns debiased image.
    """
    nonzero_mask = (image > 0)
    deb_image = np.zeros(image.shape)

    image_nz = image[nonzero_mask]
    B_nz = params.B[nonzero_mask]

    deb_image[nonzero_mask] = np.exp(np.log(image_nz) - B_nz)
    return deb_image


def normalize(image: Array[float, ('M', '...')], n_seqs: int, 
              params: ParameterEstimate, target_intensity: float = 1000., 
              norm_fn: Callable = None,  per_seq_norm: bool = True):
    """Normalizes image intensity values with LapGM parameter information.

    Normalization procedure depends on chosen norm_fn. By default this
    will be a max normalization using the largest computed mean.

    Args:
        image: 'M' channel image with variable spatial dimensions.
        n_seqs: Specifies number of sequences 'M' in multichannel image.
        params: Previously computed LapGM parameters.
        target_intensity: Scale to target when normalizing image intensities.
        norm_fn: Function to use when normalizing image. Takes in: 
            multichannel + spatial Array, ParameterEstimate, and an optional
            sequence specifier index.
        per_seq_norm: Normalizes different sequences independently if true.

    Returns normalized image with target scaling applied.
    """
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
    """Normalizes image by taken largest parameter mean and dividing through.

    For multichannel concurrent normalization, the largest mean of each
    channel is taken and balanced through a least squares procedure.

    Args:
        image: 'M' channel image with variable spatial dimensions.
        params: Previously computed LapGM parameters.
        seq_id: Specifies which image sequence to normalize against. If none,
            multichannel normalization routine will be used.
    """
    mu = params.mu
    if seq_id is None:
        mu_mx = np.exp(np.max(mu, axis=1))

        # Least-squares estimate on TARGT * [1,...,1]
        sc = (mu_mx @ mu_mx) / np.sum(mu_mx)
    else:
        sc = np.exp(np.max(mu[:,seq_id]))

    return sc
