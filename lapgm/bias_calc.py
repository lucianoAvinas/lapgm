import random 

from lapgm.typing_utils import Array
from lapgm.param_utils import ParameterEstimate

# CPU specification on default
import numpy as ap
import scipy.sparse as sp
from scipy.stats import multivariate_normal as multi_normal
_USE_GPU = False


def set_compute(assign_bool: bool):
    """Set preferred array package"""
    global _USE_GPU, ap, sp, multi_normal
    
    if assign_bool:
        import cupy as ap
        import cupyx.scipy.sparse as sp
        from cupyx_multivariate_normal import multivariate_normal as multi_normal
        _USE_GPU = True

    else:
        import numpy as ap
        import scipy.sparse as sp
        from scipy.stats import multivariate_normal as multi_normal
        _USE_GPU = False


def compute_bias_field(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
                       params: ParameterEstimate, bias_tol: float, max_iters: int, 
                       random_seed: int, print_tols: bool) -> ParameterEstimate:
    """Computes bias field through maximum a poseriori estimation with EM-like updates.

        Updates are done through a randomly permuted block cyclic ascent. All spatial
        arrays are flattened to vector form.

    Args:
        I_log (Array[float, ('M', 'N')]): Multichannel log-image with 'M' channels and 'N'
            voxels in flattened dimenions.
        params (ParameterEstimate): Contains initial parameter estimate and parameter history.
        bias_tol (float): A tolerance value on the between iteration difference of bias fields
            estimates. Computation is stopped for difference less than tolerance.
        max_iters (int): Maximum number of complete EM steps to do.
        random_seed (int): Value to seed steps in the random permuted block ascent.
        print_tols (bool): Whether to print bias field relative differences between updates.

    Returns:
        ParameterEstimate:
            Final parameter estimate with parameter history.
    """
    func_inds = list(range(3))
    phases = [e_step, gauss_step, bias_step]

    last_ind = 2
    random.seed(random_seed)
    while params.Bdiff > bias_tol and t < max_iters:
        for ind in func_inds:
            step_funcs[ind](I_log, L, params)

        if print_tols:
            print(f'iter: {t}, Bdiff: {params.Bdiff}')

        random.shuffle(func_inds)
        if func_inds[0] == last_ind:
            func_inds = func_inds[::-1]
        last_ind = func_inds[-1]

        t += 1

    return params


def e_step(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
           params: ParameterEstimate) -> None:
    """Expectation step. Updates posterior class probabilies 'w'.

    Args:
        I_log (Array[float, ('M', 'N')]): Multichannel log-image with 'M' channels and 'N'
            voxels in flattened dimenions.
        params (ParameterEstimate): Contains initial parameter estimate and parameter history.

    Returns:
        None
    """
    N = I.shape[1]
    K = params.n_classes

    w = ap.zeros((K, N))
    I_BT = (I_log - params.B[ap.newaxis]).T
    for k in range(K):
        w[k] = params.pi[k] * multi_normal.pdf(I_BT, params.mu[k], params.Sigma[k],
                                               allow_singular=True)
    w = w/w.sum(axis=0)

    params.save('w', w)


def gauss_step(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
               params: ParameterEstimate) -> None:
    """Gaussian step. Updates gaussian parameters 'pi', 'mu', and 'Sigma'.

    Args:
        I_log (Array[float, ('M', 'N')]): Multichannel log-image with 'M' channels and 'N'
            voxels in flattened dimenions.
        params (ParameterEstimate): Contains initial parameter estimate and parameter history.

    Returns:
        None
    """
    N = I.shape[1]
    K = params.n_classes

    w_sum = params.w.sum(axis=1)
    pi = w_sum / N

    resid1 = I_log - params.B[ap.newaxis]
    mu = params.w @ resid1.T / w_sum[:,ap.newaxis]

    Sigma = ap.zeros(params.Sigma.shape)

    for k in range(K):
        resid2 = resid1 - params.mu[k][:,ap.newaxis]
        Sigma[k] = (resid2 * params.w[k]) @ resid2.T / w_sum[k]

    params.save('pi', pi)
    params.save('mu', mu)
    params.save('Sigma', Sigma)


def bias_step(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
              params: ParameterEstimate) -> None:
    """Bias step. Updates the bias field 'B' and optionally the regularization 'tau'.

    Args:
        I_log (Array[float, ('M', 'N')]): Multichannel log-image with 'M' channels and 'N'
            voxels in flattened dimenions.
        params (ParameterEstimate): Contains initial parameter estimate and parameter history.

    Returns:
        None
    """
    M,N = I.shape
    K = params.n_classes

    w = params.w
    ones_vec = ap.ones(M)

    B_sum = ap.zeros((N,))
    B_inv = ap.zeros((N,))
    for k in range(K):
        ones_preck = ap.linalg.solve(params.Sigma[k], ones_vec)
        w_demean = w[k] * (I_log - params.mu[k][:,ap.newaxis])

        B_sum += ones_preck @ w_demean
        B_inv += w[k] * (ones_vec @ ones_preck)

    L_aug = L + sp.diags(B_inv)
    Bhat = params.solver(L_aug, B_sum)

    B_prev = params.B
    prev_norm = ap.linalg.norm(B_prev)
    Bdiff = ap.inf if prev_norm  == 0 else ap.linalg.norm(Bhat - B_prev)/prev_norm 

    params.save('B', Bhat)
    params.save('Bdiff', Bdiff)
