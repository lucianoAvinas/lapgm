from operator import mul
from functools import reduce
from sklearn.cluster import KMeans

# CPU specification on default
import numpy as ap
import scipy.sparse as sp
_USE_GPU = False

from lapgm.typing_utils import Array


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


class ParameterEstimate:
    """Class to hold and transfer computed parameter estimates.

    Attributes:
        spat_shape (tuple): Shape of spatial dimensions for image.
        n_classes (int): Number of Gaussian classes 'K'.
        pi (Array[float, 'K']): Array of class probabilities.
        mu (Array[float, ('K', 'M')]): Multi-sequence means indexed by class. Shape 'M' 
            determines the number of sequences.       
        Sigma (Array[float, ('K', 'M', 'M')]): Multi-sequence covariances indexed by class.
        w (Union[ArrayGPU['K,N', float], ArrayGPU['K,...', float]]: Array of class
            posterior probabilities at each voxel. May be flattend or with spatial
            dimensions. <Shape>'N' is the total number of voxels.
        B (Union[ArrayGPU['N', float], ArrayGPU['...', float]]): Estimated bias field.
            May be flattend or with spatial dimensions. <Shape>'N' is the total number of
            voxels.
        tau (float): Prior determining regularization strength of Laplacian penalty.
        gamma (float), default=1: Hyperprior suggesting how far 'tau' should be from 0.
        Bdiff (float), default=np.inf: Relative difference between the last bias field
            estimates.
    """
    def __new__(cls, params_obj, image, log_image, init_settings, solver_settings, save_settings):
        if isinstance(params_obj, cls):
            obj = params_obj
        else:
            obj = object.__new__(cls)
            obj.initialize_parameters(image, log_image, init_settings)
            obj.setup_solver(None, solver_settings)
            obj.setup_saving(save_settings)

        return obj

    def initialize_parameters(self, image, log_image, init_settings):
        if init_settings['log_initialize']:
            w = get_class_masks(log_image, init_settings['n_classes'], init_settings['n_init'],
                                init_settings['max_iters'])

        else:
            w = get_class_masks(image, init_settings['n_classes'], init_settings['n_init'],
                                init_settings['max_iters'])

        pi, mu, Sigma = init_gaussian_mixture(log_image, w)

        self.w = w
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma

        self.B = ap.zeros(image.shape[1:])

    def setup_solver(self, solver, solver_settings):
        if solver is None:
            solver = sp.linalg.cg

        self.solver = lambda A,b: solver(A, b, **solver_settings)


    def setup_saving(self, save_settings):
        self.store_history = save_settings['store_history']

        if save_settings['unstore_large']:
            self.large_attrs = ('w', 'B')
        else:
            self.large_attrs = tuple()

        self.Bdiff = None

        self.w_h = []
        self.pi_h = []
        self.mu_h = []
        self.Sigma_h = []

        self.B_h = []
        self.Bdiff_h = []

    def save(self, attr_name, attr):
        if self.store_history and attr_name not in self.large_attrs:
            getattr(self, f'{attr_name}_h').append(attr)

    def upscale_and_offload_parameters(self, upscaler, ds_spat, scale_fctr):
        # number of spatial dimensions
        dims = len(ds_spat)

        # downscale full spatial shapes
        B_ds = ds_spat
        w_ds = [self.w.shape[0]] + list(ds_spat)

        # scale factors to use when upscaling
        B_scf = [scale_fctr]*dims
        w_scf = [1] + [scale_fctr]*dims

        # per parameter scaling functions
        B_scaler = lambda B: self.offload(upscaler(B.reshape(B_ds), B_scf))
        w_scaler = lambda w: self.offload(renormalize_probs(upscaler(w.reshape(w_ds), w_scf)))

        # scale and offload last estimated parameters
        self.B = B_scaler(self.B)
        self.w = w_scaler(self.w)
        self.pi = self.offload(self.pi)
        self.mu = self.offload(self.mu)
        self.Sigma = self.offload(self.Sigma)

        # offload parameter histories. will not apply scaling.
        for i in range(len(self.B_h)):
            self.B_h[i] = self.offload(self.B_h[i])
            self.w_h[i] = self.offload(self.w_h[i])
            self.pi_h[i] = self.offload(self.pi_h[i])
            self.mu_h[i] = self.offload(self.mu_h[i])
            self.Sigma_h[i] = self.offload(self.Sigma_h[i])

    def offload(self, x):
        if _USE_GPU:
            ap.asnumpy(x)
        return x

    def get_estimate(self):
        return dict(log_B=self.B, w=self.w, pi=self.pi, log_mu=self.mu, log_Sigma=self.Sigma)

    def get_history(self):
        return dict(log_B=self.B_h, logB_diff=self.Bdiff_h, w=self.w_h, pi=self.pi_h, 
                    log_mu=self.mu_h, log_Sigma=self.Sigma_h)
        

def renormalize_probs(prob_arr, class_ax=0):
    # find smallest contributions per array index
    arr_mins = ap.min(prob_arr, axis=class_ax)

    # shift negative contributions up
    prob_arr = prob_arr - arr_mins * (arr_mins < 0)
    
    # renormalize
    prob_arr = prob_arr / np.sum(prob_arr, axis=class_ax)

    return prob_arr


def get_class_masks(I: Array[float, ('M', '...')], n_classes: int, n_init: int, 
                    max_iters : int) -> Array[bool, ('K', '...')]:
    """Returns class membership arrays depending on closest cluster.

        Class membership is determined using the k-means algorithm on the
        spatially-flattened, multichannel image.

    Args:
        I (Array[float, ('M', '...')]): Multichannel image with 'M' sequences and
            arbitrary number of spatial dimensions.
        n_classes (int): Number of classes 'K' to fit on the spatial-flattened
            multichannel data.
        n_init (int), default=10:
            Number of times k-means runs with a different centroid seed. Best output of 
            n_init is used.
        max_iter (int), default=300:
            Number of maximum iterations per k-means run.

    Returns:
        Array[bool, ('K', '...')]:
            Returns a boolean array with same spatial dimensions as I. First axis
            contains class membership.
    """
    # k-means method uses numpy arrays so cast appropriately
    if _USE_GPU:
        I = ap.asnumpy(I)

    _, N = I.shape
    labels = KMeans(n_clusters=n_classes, n_init=n_init, max_iter=max_iters).fit(
                    I.T).labels_
    
    # one-hot encoding of labels
    label_masks = ap.zeros((n_classes, N), dtype=bool)
    label_masks[labels, ap.arange(N)] = True

    return label_masks


def init_gaussian_mixture(I_log: Array[float, ('M', '...')], w: Array[bool, ('K', '...')]) -> \
                          ParameterEstimate:
    """Calculates initial parameters using a previously computed class mask.

    Args:
        I_log (Array[float, ('M', '...')]): Multichannel log-image with 'M' sequences and
            arbitrary number of spatial dimensions.
        w (Array[bool, ('K', '...')]): Hard label class for each entry of I.

    Returns:
        ParameterEstimate:
            Initial parameter estimate.
    """
    M, N = I.shape
    K, _ = w.shape

    pi = ap.mean(w, axis=1)

    mu = ap.zeros((K, M))
    Sigma = ap.zeros((K, M, M))

    for k in range(K):
        Ilog_k = I_log[:, w[k]]
        mu[k] = Ilog_k.mean(axis=1)
        Sigma[k] = np.cov(Ilog_k)

    return pi, mu, Sigma
    