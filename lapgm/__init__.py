import numpy as np
from typing import Callable

from lapgm.config import _CUPY_PATH
from lapgm.typing_utils import Array
from lapgm.lapgm_estim import check_seq_shape
from lapgm.param_setup import ParameterEstimate

from lapgm.lapgm_estim import set_compute as le_setcomp
from lapgm.bias_calc import set_compute as bc_setcomp
from lapgm.param_setup import set_compute as ps_setcom
from lapgm.laplacian_routines import set_compute as lr_setcomp



def use_gpu(assign_bool: bool):
    """Propagate GPU settings to relevant modules"""
    if config._CUPY_PATH is None:
            raise ModuleNotFoundError('CuPy array package not found')

    le_setcomp(assign_bool)
    bc_setcomp(assign_bool)
    pu_setcomp(assign_bool)
    lr_setcomp(assign_bool)


def debias(image: Array[float, ('M', '...')], params: type[ParameterEstimate]):
    nonzero_mask = (image > 0)
    deb_image = np.zeros(image.shape)

    deb_image[nonzero_mask] = np.exp(np.log(image[nonzero_mask]) - params.B)
    return deb_image


def normalize(image: Array[float, ('M', '...')], n_seqs: int, 
              params: type[ParameterEstimate], target_intensity: float = 1000., 
              norm_fn: Callable = None,  per_seq_norm: bool = True):

    image = check_seq_shape(image, n_seqs)
    norm_image = np.zeros(image.shape)

    if norm_fn is None:
        norm_fn = max_norm_fn
    
    if per_seq_norm:
        for i in range(n_seqs):
            norm_image[i] = norm_fn(image[i], params, i) * target_intensity
    else:
        norm_image = norm_fn(image, params) * target_intensity

    return norm_image


def max_norm_fn(image: Array[float, ('M', '...')], params: type[ParameterEstimate],
                seq_id: int = None):

    seq_id = 0 if seq_id is None else seq_id
    return image / np.exp(np.max(params.mu[:,seq_id]))
