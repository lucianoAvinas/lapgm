import debias
import bias_calc
import param_utils
import laplacian_routines

from config import cupy_path


def use_gpu(assign_bool: bool):
    """Propagate GPU settings to relevant modules"""
    if cupy_path is None:
            raise ModuleNotFoundError('CuPy array package not found')

    debias.set_compute(assign_bool)
    bias_calc.set_compute(assign_bool)
    param_utils.set_compute(assign_bool)
    laplacian_routines.set_compute(assign_bool)
