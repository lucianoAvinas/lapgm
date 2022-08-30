from __future__ import annotations

from typing import Callable, Any
from .typing_details import Array
from sklearn.cluster import KMeans

### Typing details to be replaced with real packages at runtime ###
from .typing_details import ArrayPackage as ap
from .typing_details import SparseLinalgPackage as spl

# Compute always initialized to CPU
_ON_GPU = False


class ParameterEstimate:
    """Stores and transfers computed parameters.

    Args:
        params_obj: A ParameterEstimate object to inherit. Use case is to allow users
            to tweak ParameterEstimate and resubmit to 'lapgm_estim.estimate_parameters'.
        image: 'M' channel image with variable spatial dimensions.
        log_image: 'M' channel log-image with 'N' voxels flattened in the last axis.
        init_setting: Settings to run on the k-means initializer.
        solver_settings: Sparse solver settings. Default solver is a conjugate gradient
            routine. Solver type can be updated through the 'setup_solver' method.
        save_settings: Determines which parameters get stored in history.


    Attributes:
        spat_shape_in (tuple): Spatial dimension of image input. Totals 'N' elements.
        spat_shape_out (tuple): Original spatial dimension of image (not input).
        n_classes (int): Number of classes 'K' to use in mixture.
        pi (Array[float, 'K']): Array of class probabilities.
        mu (Array[float, ('K', 'M')]): Multi-sequence means.
        Sigma (Array[float, ('K', 'M', 'M')]): Multi-sequence covariances.
        w (Array[float, ('K', 'N')]): Array of class posterior probabilities.
        B (Array[float, N']): Bias field estimate of the image.
        Bdiff (float): Relative difference between subsequent bias field estimates.
        history (dict): Parameter estimate history for previous iterations. Parameter
            saving is determined by the 'save_settings' argument.
    """

    def __new__(cls, params_obj: ParameterEstimate, image: Array[float, ('M','...')], 
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
        """Initialize Gaussian parameters through a k-means method.

        Args:
            image: 'M' channel image with variable spatial dimensions.
            log_image: 'M' channel log-image with 'N' flattened voxels.
            init_setting: Settings to run on the k-means initializer.
        """
        n_seqs, *spat_shape = image.shape

        if init_settings['log_initialize']:
            w = get_class_masks(log_image, init_settings['kmeans_settings'])
        else:
            # flatten spatial axes before k-means init
            image = image.reshape(n_seqs, -1)

            w = get_class_masks(image, init_settings['kmeans_settings'])

        pi, mu, Sigma = init_gaussian_mixture(log_image, w)

        self.spat_shape_in = spat_shape
        self.spat_shape_out = None

        self.n_seqs = n_seqs
        self.n_classes = pi.shape[0]

        self.w = w.astype(float)
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma

        self.B = ap.zeros(log_image.shape[1])
        self.Bdiff = ap.inf

    def setup_solver(self, solver: Callable, solver_settings: dict):
        """Setup sparse linear solver with optional keyword settings. 

        By default the solver of choice is a conjugate gradient method.

        Args:
            solver: Sparse linear solver.
            solver_settings: Sparse solver settings.
        """
        if solver is None:
            self.solver = sp_cg_wrapper(solver_settings)
        else:
            self.solver = lambda A,b: solver(A, b, **solver_settings)

    def setup_saving(self, save_settings: dict):
        """Setup how parameter history should be saved."""
        self.store_history = save_settings['store_history']

        if save_settings['unstore_large']:
            self.large_attrs = ('w', 'B')
        else:
            self.large_attrs = tuple()

        self.history = dict(w=[], pi=[], mu=[], Sigma=[], B=[], Bdiff=[])

    def save(self, attr_name: str, attr: Any):
        """Save parameter by name and append to history if valid."""
        setattr(self, attr_name, attr)
        if self.store_history and attr_name not in self.large_attrs:
            self.history[attr_name].append(attr)

    def upscale_parameters(self, upscaler: Callable, orig_spat: tuple[int], 
                           scale_fctr: float):
        """Upscales parameters with image dimensions while offload any GPU params.

        Args:
            upscaler: Function which upscales downscaled image to original shape.
            orig_spat: Original spatial shape to match.
            scale_fctr: Scale factor to upscale spatial dimensions by.
        """
        # spatial dimension of outgoing parameters
        self.spat_shape_out = orig_spat

        # number of spatial dimensions
        dims = len(self.spat_shape_in)

        # allows for image upscaling on arrays with prepended information
        upscale_wrapper = lambda dat, pre_dim: upscaler(dat, pre_dim + list(orig_spat), 
                                               [1]*len(pre_dim) + [scale_fctr]*dims)

        # reshape and scale parameters that have image dimensions
        self.B = self.B.reshape(self.spat_shape_in)
        self.B = upscale_wrapper(self.B, [])

        K = self.w.shape[0]
        self.w = self.w.reshape([K] + list(self.spat_shape_in))
        self.w = renormalize_probs(upscale_wrapper(self.w, [K]))

        # offload parameters for CPU use
        for param_nm in ('w', 'pi', 'mu', 'Sigma', 'B', 'Bdiff'):
            self.offload_param_and_history(param_nm)

    def offload_param_and_history(self, param_nm: str):
        """Cast parameter and its history to CPU."""
        if _ON_GPU:
            # offload parameter estimate
            setattr(self, param_nm, ap.asnumpy(getattr(self, param_nm)))

            # offload parameter history
            self.history[param_nm] = [ap.asnumpy(val_i) for val_i in self.history[param_nm]]
     

def sp_cg_wrapper(solver_settings: dict):
    """Wraps conjugate gradient method to return estimate with no extra info."""
    return lambda A,b: spl.cg(A, b, **solver_settings)[0]


def renormalize_probs(prob_arr: Array[float, ('K','...')], class_ax: int = 0):
    """Renormalize probability array. 
    
    Will shift up negative contributions to enforce non-negativity.
    
    Args:
        prob_arr: Array of probabilities.
        class_ax: Axis which class information for probabilities is located.

    Returns normalized probability array.
    """
    # find smallest contributions per array index
    arr_mins = ap.min(prob_arr, axis=class_ax)

    # shift negative contributions up
    prob_arr = prob_arr - arr_mins * (arr_mins < 0)
    
    # renormalize
    prob_arr = prob_arr / ap.sum(prob_arr, axis=class_ax)

    return prob_arr


def get_class_masks(I: Array[float, ('M', '...')], kmeans_settings: dict):
    """Returns class membership arrays depending on closest cluster.

        Class membership is determined using the k-means algorithm on the spatially-flattened, 
        multichannel image.

    Args:
        I: 'M' channel image with 'N' voxels flattened in the last axis.
        kmeans_settings: Keyword arguments for sklearn.cluster.KMeans.

    Returns a boolean array. Each spatial position is assigned a one-hot class encoding.
    """
    # sklearn's k-means uses numpy arrays, so cast appropriately
    if _ON_GPU:
        I = ap.asnumpy(I)

    _, N = I.shape
    n_clusters = kmeans_settings['n_clusters']
    labels = KMeans(**kmeans_settings).fit(I.T).labels_
    
    # one-hot encoding of labels
    label_masks = ap.zeros((n_clusters, N), dtype=bool)
    label_masks[labels, ap.arange(N)] = True

    return label_masks


def init_gaussian_mixture(I_log: Array[float, ('M', '...')], w: Array[bool, ('K', '...')]):
    """Uses a class label mask to estimate Gaussian mixture parameters.

    Gaussian parameters are multivariate with respect to the number of channels 'M' of the
    image.

    Args:
        I_log: log_image: 'M' channel log-image with 'N' voxels flattened in the last axis.
        w: Hard label class for each entry of I_log.

    Returns mixture class probabilities, Gaussian means, and Gaussian covariances.
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
    