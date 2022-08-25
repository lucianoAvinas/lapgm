import numpy as np
from typing import Callable

from lapgm.config import _CUPY_PATH
from lapgm.typing_utils import Array
from lapgm.param_setup import ParameterEstimate
from lapgm.lapgm_estim import LapGM, check_seq_shape
from lapgm.view_utils import view_center_slices, view_class_map, \
                             view_distributions

from lapgm.lapgm_estim import set_compute as le_setcomp
from lapgm.bias_calc import set_compute as bc_setcomp
from lapgm.param_setup import set_compute as ps_setcomp
from lapgm.laplacian_routines import set_compute as lr_setcomp


def use_gpu(assign_bool: bool):
    """Propagate GPU settings to relevant modules"""
    if config._CUPY_PATH is None:
            raise ModuleNotFoundError('CuPy array package not found')

    le_setcomp(assign_bool)
    bc_setcomp(assign_bool)
    ps_setcomp(assign_bool)
    lr_setcomp(assign_bool)


def debias(image: Array[float, ('M', '...')], params: type[ParameterEstimate]):
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


def max_norm_fn(image: Array[float, ('M', '...')], params: ParameterEstimate, seq_id: int = None):
    mu = params.mu

    if seq_id is None:
        mu_mx = np.exp(np.max(mu, axis=1))
        sc = (mu_mx @ mu_mx) / np.sum(mu_mx)  # Least-squares estimate on [TARGT,...,TARGT]
    else:
        sc = np.exp(np.max(mu[:,seq_id]))

    return sc
