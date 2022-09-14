from .typing_details import Array
from .param_setup import ParameterEstimate

### Typing details to be replaced with real packages at runtime ###
from .typing_details import ArrayPackage as ap
from .typing_details import SparsePackage as sp
from .typing_details import RuntimeFunc as multi_normal


def compute_bias_field(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
                       params: ParameterEstimate, bias_tol: float, max_iters: int, 
                       print_tols: bool):
    """Computes relevant LapGM parameters through a MAP probability estimate.

    Interleaves expectation steps with Gaussian and bias steps. Major dimensions 
    are: 'M' sequences, 'K' classes, and 'N' spatial elements. Spatial elements 
    may belong to a general d-dimensional grid but are flattened during optimization.

    Args:
        I_log: 'M' channel log-image with 'N' voxels flattened in the last axis.
        L: Weighted Laplacian matrix for spatial structure of the 'N' nodes.
        params: Container for parameter estimate and history.
        bias_tol: Relative tolerance to stop on if subsequent bias estimates are 
            close in value.
        max_iters: Maximum number of MAP optimization steps to do.
        print_tols: Prints relative bias differences if true.

    Returns final parameter estimate with parameter history.
    """
    step_fns = [e_step, gauss_step, bias_step]

    func_inds = [0,1,0,2]

    t = 0
    while params.Bdiff > bias_tol and t < max_iters:
        for ind in func_inds:
            step_fns[ind](I_log, L, params)

        if print_tols:
            print(f'iter: {t}, Bdiff: {params.Bdiff}')

        t += 1

    return params


def e_step(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
           params: ParameterEstimate):
    """Update step for posterior class probabilies 'w'.

    Args:
        I_log: 'M' channel log-image with 'N' voxels flattened in the last axis.
        L: Weighted Laplacian matrix for spatial structure of the 'N' nodes.
        params: Container for parameter estimate and history.
    """
    N = I_log.shape[1]
    K = params.n_classes

    w = ap.zeros((K, N))
    I_BT = (I_log - params.B[ap.newaxis]).T
    for k in range(K):
        w[k] = params.pi[k] * multi_normal.pdf(I_BT, params.mu[k], params.Sigma[k],
                                               allow_singular=True)
    w = w/w.sum(axis=0)

    params.save('w', w)


def gauss_step(I_log: Array[float, ('M', 'N')], L: Array[float, ('N', 'N')], 
               params: ParameterEstimate):
    """Update step for Gaussian parameters 'pi', 'mu', and 'Sigma'.

    Args:
        I_log: 'M' channel log-image with 'N' voxels flattened in the last axis.
        L: Weighted Laplacian matrix for spatial structure of the 'N' nodes.
        params: Container for parameter estimate and history.
    """
    N = I_log.shape[1]
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
              params: ParameterEstimate):
    """Update step for log bias 'B'.

    Args:
        I_log: 'M' channel log-image with 'N' voxels flattened in the last axis.
        L: Weighted Laplacian matrix for spatial structure of the 'N' nodes.
        params: Container for parameter estimate and history.
    """
    M,N = I_log.shape
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
