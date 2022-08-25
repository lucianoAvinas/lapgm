from typing import Callable, Any
from sklearn.cluster import KMeans

# CPU specification on default
import numpy as ap
import scipy.sparse.linalg as spl
_USE_GPU = False

from lapgm.typing_utils import Array


def set_compute(assign_bool: bool):
    """Set preferred array package"""
    global _USE_GPU, ap, spl
    
    if assign_bool:
        import cupy as ap
        import cupyx.scipy.sparse.linalg as spl
        _USE_GPU = True

    else:
        import numpy as ap
        import scipy.sparse.linalg as spl
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

    def __new__(cls, params_obj: object, image: Array[float, ('M','...')], 
                log_image: Array[float, ('M', 'N')], init_settings: dict, 
                solver_settings: dict, save_settings: dict):
        if isinstance(params_obj, cls):
            obj = params_obj
        else:
            obj = object.__new__(cls)
            obj.initialize_parameters(image, log_image, init_settings)
            obj.setup_solver(None, solver_settings)
            obj.setup_saving(save_settings)

        return obj

    def initialize_parameters(self, image: Array[float, ('M','...')], 
                              log_image: Array[float, ('M', 'N')], 
                              init_settings: dict):
        n_seqs, *spat_shape = image.shape

        if init_settings['log_initialize']:
            w = get_class_masks(log_image, init_settings['n_classes'], 
                                init_settings['n_init'],
                                init_settings['max_iters'])
        else:
            # flatten spatial axes before k-means init
            image = image.reshape(n_seqs, -1)

            w = get_class_masks(image, init_settings['n_classes'], 
                                init_settings['n_init'],
                                init_settings['max_iters'])

        pi, mu, Sigma = init_gaussian_mixture(log_image, w)

        self.n_seqs = n_seqs
        self.spat_shape = spat_shape
        self.n_classes = pi.shape[0]

        self.w = w
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma

        self.B = ap.zeros(log_image.shape[1])

    def setup_solver(self, solver: Callable, solver_settings: dict):
        if solver is None:
            self.solver = sp_cg_wrapper(solver_settings)
        else:
            self.solver = lambda A,b: solver(A, b, **solver_settings)

    def setup_saving(self, save_settings: dict):
        self.store_history = save_settings['store_history']

        if save_settings['unstore_large']:
            self.large_attrs = ('w', 'B')
        else:
            self.large_attrs = tuple()

        self.Bdiff = ap.inf

        self.w_h = []
        self.pi_h = []
        self.mu_h = []
        self.Sigma_h = []

        self.B_h = []
        self.Bdiff_h = []

    def save(self, attr_name: str, attr: Any):
        setattr(self, attr_name, attr)
        if self.store_history and attr_name not in self.large_attrs:
            getattr(self, f'{attr_name}_h').append(attr)

    def upscale_parameters(self, upscaler: Callable, orig_spat: tuple[int], 
                           scale_fctr: float):
        # number of spatial dimensions
        dims = len(self.spat_shape)

        # allows for image upscaling on arrays with prepended information
        upscale_wrapper = lambda dat, pre_dim: upscaler(dat, pre_dim + list(orig_spat), 
                                               [1]*len(pre_dim) + [scale_fctr]*dims)

        # reshape and scale parameters that have image dimensions
        self.B = self.B.reshape(self.spat_shape)
        self.B = upscale_wrapper(self.B, [])

        K = self.w.shape[0]
        self.w = self.w.reshape([K] + list(self.spat_shape))
        self.w = renormalize_probs(upscale_wrapper(self.w, [K]))

        # offload parameters for CPU use
        for param_nm in ('w', 'pi', 'mu', 'Sigma', 'B', 'Bdiff'):
            self.offload_param_and_history(param_nm)

    def offload_param_and_history(self, param_nm: str):
        if _USE_GPU:
            val = getattr(self, param_nm)
            setattr(self, param_nm, ap.asnumpy(val))

            val = getattr(self, f'{param_nm}_h')
            setattr(self, f'{param_nm}_h', [ap.asnumpy(val_i) for val_i in val])
     

def sp_cg_wrapper(solver_settings: dict):
    return lambda A,b: spl.cg(A, b, **solver_settings)[0]


def renormalize_probs(prob_arr: Array[float, ('K','...')], class_ax: int = 0):
    # find smallest contributions per array index
    arr_mins = ap.min(prob_arr, axis=class_ax)

    # shift negative contributions up
    prob_arr = prob_arr - arr_mins * (arr_mins < 0)
    
    # renormalize
    prob_arr = prob_arr / ap.sum(prob_arr, axis=class_ax)

    return prob_arr


def get_class_masks(I: Array[float, ('M', '...')], n_classes: int, n_init: int, max_iters : int):
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


def init_gaussian_mixture(I_log: Array[float, ('M', '...')], w: Array[bool, ('K', '...')]):
    """Calculates initial parameters using a previously computed class mask.

    Args:
        I_log (Array[float, ('M', '...')]): Multichannel log-image with 'M' sequences and
            arbitrary number of spatial dimensions.
        w (Array[bool, ('K', '...')]): Hard label class for each entry of I.

    Returns:
        ParameterEstimate:
            Initial parameter estimate.
    """
    M, N = I_log.shape
    K, _ = w.shape

    pi = ap.mean(w, axis=1)

    mu = ap.zeros((K, M))
    Sigma = ap.zeros((K, M, M))

    for k in range(K):
        Ilog_k = I_log[:, w[k]]
        mu[k] = Ilog_k.mean(axis=1)
        Sigma[k] = ap.cov(Ilog_k)

    return pi, mu, Sigma
    