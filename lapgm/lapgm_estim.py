from typing import Callable, Any

from .typing_details import Array
from .bias_calc import compute_bias_field
from .param_setup import ParameterEstimate
from .laplacian_routines import prepare_wgts, hyper_ellipsoid_radius, \
                                construct_dirichlet_lap, weight_laplacian

### Typing details to be replaced with real packages at runtime ###
from .typing_details import ArrayPackage as ap
from .typing_details import RuntimeFunc as zoom


class LapGM:
    """Sets up LapGM debias method and runs parameter estimates for multichannel images.

    Args:
        downscale_factor: Whole number to downsample spatial dimensions by.
        scaling_order: Interpolation smoothness order to use when upscaling.
        store_history: Determines whether previous parameter estimates should be recorded.
        unstore_large: Stops any recording on large spatial parameters like 'w' (posterior
            class probabilities) and 'B' (bias field estimate).

    Attributes:
        downscale_factor (int): Whole number to downsample spatial dimensions by.
        scaling_order (int): Interpolation smoothness order to use when up/downscaling.
        save_settings (dict[str, bool]): Collects store_history and unstore_large.
        tau (float): Hyperparameter penalty strength on bias field gradient
        init_settings (dict): Settings to pass on to a k-means initializer.
        weight_settings (dict): Settings to determine weighting scheme on graph Laplacian.
    """
    def __init__(self, downscale_factor: int = 1, scaling_order: int = 3, 
                 store_history: bool = False, unstore_large: bool = True):
        if downscale_factor < 1:
            raise ValueError('Downscale factor should be a whole number.')

        self.downscale_factor = downscale_factor
        self.scaling_order = scaling_order
        self.save_settings = dict(store_history=store_history, unstore_large=unstore_large)

        self.tau = None
        self.init_settings = None
        self.weight_settings = None

    def specify_cylindrical_decay(self, alpha: float, axes_of_symmetry: tuple[int] = (0,), 
                                  center: tuple[int] = (0,0), semi_axes: tuple[int] = (1,1)):
        """Sets a power-decay weighting with cylindrical symmetry on Laplacian 'L'

        Args:
            alpha: Positive power to decay by. Goes as x^(-alpha).
            axes_of_symmetry: Spatial axes on which weighting should remain invariant.
            center: Center point for outgoing radial weights of cylinder. Center is in image
                centered coordinates, so +/-1 moves center by one unit. 
            semi_axes: Semi-major axes of cylinder. 
        """
        wgt_func = lambda x: x**(-alpha)
        self.weight_settings = dict(wgt_func=wgt_func, axes_of_symmetry=axes_of_symmetry, 
                                    center=center, semi_axes=semi_axes)

    def set_hyperparameters(self, tau: float, n_classes: int, log_initialize: bool = True, 
                            kmeans_n_init: int = 5, kmeans_max_iters: int = 300):
        """Sets hyperparameters for the LapGM method.

        Initializer settings contains k-means setup information.

        Args:
            tau: Inverse regulariztion strength for bias field gradient
            log_initialize: Determines whether raw intensity or log intensities should be
                used when running k-means initializer.
            n_classes: Number of classes for the LapGM model.
            kmeans_n_init: Number of times to run k-means with different centroid seeds.
            kmeans_max_iters: Maximum number of iterations for a given k-means run.
        """
        self.tau = tau
        self.init_settings = dict(log_initialize=log_initialize, kmeans_settings=dict(
                                  n_clusters=n_classes, n_init=kmeans_n_init, 
                                  max_iter=kmeans_max_iters, random_state=None))

    def estimate_parameters(self, image: Array[float, ('M', '...')], 
                            bias_tol: float = 1e-4, max_em_iters: int = 25, 
                            max_cg_iters: int = 25000, random_seed: int = 1,  
                            print_tols: bool = False, params_obj: ParameterEstimate = None, 
                            custom_wgts: Array[float, 'N'] = None):
        """Estimate parameters for the LapGM model.

        Setting max_em_iters to zero returns default initialized ParameterEstimate object 
        for image input.

        Args:
            image: Multichannel image with variable spatial dimensions. Contains sequence
                metadata from __init__.to_sequence_array.
            bias_tol: Relative tolerance to stop on if subsequent bias estimates are 
                close in value.
            max_em_iters: Maximum number of MAP optimization steps to do.
            max_cg_iters: Maximum number of conjugate gradient steps for bias inverse step.
            random_seed: Sets seed for the parameter estimation.
            print_tols: Prints relative bias differences if true.
            param_obj: Custom ParameterEstimate object to initialize bias field estimation.
                May contain custom solver for bias inverse step.
            custom_wgts: Custom multiplicative weights to construct Laplacian 'L' with.

        Returns a ParameterEstimate object for image input.
        """
        # check if number of sequences matches first axis length
        n_seqs = check_seq_data(image)

        # cast to relevant array
        image = ap.asarray(image)  

        # dimensions of the spatial axes
        spat_shape = image.shape[1:]

        # downscale image without straining memory
        ds_slices = [slice(None,None)] + [slice(None, None, self.downscale_factor)] * len(spat_shape)
        image = image[tuple(ds_slices)]

        # flatten spatial axes and log
        spat_ds_shape = image.shape[1:]
        log_image = fill_zeros_and_log(image).reshape(n_seqs, -1)

        # construct initial parameter estimate
        self.init_settings['kmeans_settings']['random_state'] = random_seed
        params = ParameterEstimate(params_obj, image, log_image, self.init_settings, 
                                   dict(maxiter=max_cg_iters), self.save_settings)

        # apply laplacian weighting routine
        if custom_wgts is not None:
            wgts = ap.asarray(custom_wgts)

        elif self.weight_settings is not None:
            coord_transf = hyper_ellipsoid_radius(self.weight_settings['center'], 
                                                  self.weight_settings['semi_axes'])
            wgts =  prepare_wgts(spat_ds_shape, self.weight_settings['wgt_func'], 
                                 coord_transf, self.weight_settings['axes_of_symmetry'])
        else:
            # uniform multiplicative weight if no weight is provided
            wgts = ap.ones(log_image.shape[1])

        # construct weighted Laplacian and scale by tau
        L = weight_laplacian(construct_dirichlet_lap(spat_ds_shape, True), wgts) / self.tau

        # calculate parameter estimates
        params = compute_bias_field(log_image, L, params, bias_tol, max_em_iters, print_tols)
        
        # define upscaler and apply
        upscaler = lambda img_ds, targ_shp, sc_lst: scale_and_pad(img_ds, targ_shp, sc_lst, 
                                                                  self.scaling_order)
        params.upscale_parameters(upscaler, spat_shape, self.downscale_factor)

        return params


def check_seq_data(image: Array[Any, ('M', '...')]):
    """Checks whether image first axis matches number of sequences.
    
    Interface with numpy metadata to check axis match. Subject to change upon changes to
    numpy.dtype.metadata.

    Raises ValueError on missing metadata or on sequence data mismatch

    Returns number of sequences contained in array.
    """

    img_meta = image.dtype.metadata
    if img_meta is None or 'n_seqs' not in img_meta:
        raise ValueError('Image missing sequence metadata. Apply to_sequence_array to ' + \
                         'array before calling function.')
    elif img_meta['n_seqs'] != image.shape[0]:
        raise ValueError('Mismatch between image first axis length and number of ' + \
                         'sequences specified.')

    return img_meta['n_seqs']


def fill_zeros_and_log(image: Array[Any, ('...')]):
    """Takes log on non-negative images.

    Zeros are handled through a uniform random fill with magnitude 
    up to half of minimum non-zero value of image.
    """
    zero_mask = (image == 0)

    min_val = ap.min(image[~zero_mask])/2
    zero_cnt = ap.sum(zero_mask).item()
    image[zero_mask] = ap.random.random(zero_cnt) * min_val

    return ap.log(image)


def scale_and_pad(scale_img: Array[float, ('...')], orig_dims: tuple[int], 
                  scale_fctrs: tuple[float], smooth_order: int):
    """Upscale previously downscaled image to its original dimensions.

    Original dimensions may not have divided nicely with scaling factor.
    Slight cropping and padding will be handed inside scale_and_pad.

    Args:
        scale_img: Downscaled image.
        orig_dim: Dimensions of original image.
        scale_fctrs: Tuple of scaling factors for per-axis upscales.
        smooth_order: Interpolation smoothness order for upscales.

    Returns upscaled image with original image dimensions.
    """
    img = zoom(scale_img, scale_fctrs, order=smooth_order, mode='nearest')

    # calculate dimension offset between upsampled image and original image
    dim_diff = tuple(o-n for o,n in zip(orig_dims, img.shape))

    # pad smaller dims
    img = ap.pad(img, tuple((0, max(dim, 0)) for dim in dim_diff), 'edge')

    # crop bigger dims
    img = img[tuple([slice(0, dim if dim < 0 else None) for dim in dim_diff])]

    return img
