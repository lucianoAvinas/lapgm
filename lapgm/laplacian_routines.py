from operator import mul
from functools import reduce

from typing import Callable, Any
from lapgm.typing_utils import Array

# CPU specification on default
import numpy as ap
import scipy.sparse as sp
_USE_GPU = False


def set_compute(assign_bool: bool):
    """Set preferred array package"""
    global _USE_GPU, ap, sp
    
    if assign_bool:
        import cupy as ap
        import cupyx.scipy.sparse as sp
        _USE_GPU = True

    else:
        import numpy as ap
        import scipy.sparse as sp
        _USE_GPU = False


def construct_dirichlet_lap(bounds: tuple[int], upper_only: bool = False):
    """Constructs a weighted discrete Laplacian for a multidimensional grid graph.

        Calculation done in terms of Kronecker sums. Assume Dirichlet boundary conditions.

    Args:
        bounds (tuple[int]): Respective dimensions of the multidimensional grid graph.
        wgts (Array['N', float]): Per voxel weighting array 

    Returns:
        Array['N,N', float]:
            Returns a weighted discrete Laplacian in flattened index notation. Shape 'N'
            is equal to the generalized volume of the grid.
    """
    N = reduce(mul, bounds)
    L = sp.csr_matrix((N,N), dtype=float)

    dims = len(bounds)
    kron_terms = [None] * dims

    for i in range(dims):
        for j in range(dims):

            # Build terms for the i-th term in Kronecker sum
            if j == i:
                d2 = -ap.ones(bounds[j]-1)

                if upper_only:
                    kron_terms[j] = sp.diags((d2,), (1,))

                else:
                    d1 = 2*ap.ones(bounds[j])
                    d1[0] = d1[-1] = 1
                    kron_terms[j] = sp.diags((d2,d1,d2), (-1,0,1))

            else:
                kron_terms[j] = sp.eye(bounds[j])

        # Contracts the i-th Kronecker product
        L += reduce(sp.kron, kron_terms)

    return L


def weight_laplacian(L_upper: Array[float, ('N', 'N')], wgts: Array[float, 'N']):
    # apply voxel weighting as an edge weight
    L_upper = L_upper.multiply(wgts)
    L = L_upper + L_upper.T

    # diagonal which normalizes outgoing edge weights
    D = sp.diags(ap.squeeze(ap.asarray(-L.sum(axis=0)))) 

    return L + D


def prepare_wgts(bounds: tuple, wgt_func: Callable, coord_transf: Callable, 
                 axes_of_symmetry: tuple[int] = None, zero_one_scale: bool = True, 
                 fill_zeros: bool = True):

    # subset bounds according to axis symmetries
    if axes_of_symmetry is None:
        axes_of_symmetry = tuple()
    else:
        # duck type into tuple
        try:
            axes_of_symmetry[0]
        except TypeError:
            axes_of_symmetry = tuple(axes_of_symmetry)

    bounds_subset = tuple(n_i for i,n_i in enumerate(bounds) if i not in axes_of_symmetry)

    # get centered, unit coordinates
    dim_max = (max(bounds_subset)-1)/2
    sparse_coords = [(ap.arange(n_i) - (n_i-1)/2) / dim_max for n_i in bounds_subset]

    # transform sparse coordinates (possibly to a new representation) 
    # and apply weighting scheme.
    wgts = wgt_func(coord_transf(sparse_coords))

    # normalizes wgts to [0,1]
    if zero_one_scale:
        wgts -= ap.min(wgts)
        wgts /= ap.max(wgts)

    elif ap.min(wgts) < 0:
        raise ValueError('Laplacian scaling weights should be positive')

    # replaces previous 0 values with smallest positive value in wgts 
    if fill_zeros:
        inplace_min_fill(wgts)

    # promote wgts to the original bounds dimension and then flatten
    wgts = ap.expand_dims(wgts, axes_of_symmetry)
    wgts = ap.broadcast_to(wgts, bounds).flatten()

    return wgts


def inplace_min_fill(x: Array[float, ('...')]):
    x[x == 0] = ap.min(x[x != 0])


def fill_on_none(x: Any, val: Any, sz: int):
    if x is None:
        x = [val]*sz
    return x


def hyper_ellipsoid_radius(centers: tuple[int], semi_axes: tuple[int], 
                           smoothout_origin: bool = True):
    def radial_calc(coords: list[Array[Any, ('...')]]):
        # coords are sparse

        d = len(coords)

        # intialize defaults to spherical with 0 origin
        co_cent = fill_on_none(centers, 0, d)
        co_semi = fill_on_none(semi_axes, 1, d)

        sc_coords = [(coords[i] - co_cent[i])**2/co_semi[i] for i in range(d)]
        radial_values = ap.sqrt(sum(sc_coords))

        if smoothout_origin:
            inplace_min_fill(radial_values)

        return radial_values

    return radial_calc
