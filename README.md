# lapgm

**lapgm** is an image correction software package primarily used for MR debiasing and normalization. This package is generalized to work with any number of spatial dimensions and assumes smooth, multiplicative image corruption which is invariant through image channels. Derivations and results for the spatially regularized Gaussian mixture can be found at [*LapGM: A Multisequence MR Bias Correction and Normalization Model.*](https://arxiv.org)

# Installation

Package **lapgm** can be installed using pip:
```
pip install lapgm
```
A CUDA accelerated version of **lapgm** is also available:
```
pip install lapgm[gpu]
```

# Examples

An example overview on how to debias and normalize with **lapgm** is provided 'image_correction.ipynb' in the 'examples' subdirectory. For data, three biased presets from the [*BrainWeb*](https://brainweb.bic.mni.mcgill.ca/) normal dataset were used. Some of the debiased and normalized results have been provided below.

T2 debiasing results:
<p align="center">
  <img width="300" src="https://github.com/lucianoAvinas/lapgm/raw/main/images/t2_biased.png">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="300" src="https://github.com/lucianoAvinas/lapgm/raw/main/images/t2_debiased.png">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>


T1 normalization results:
<p align="center">
  <img width="350" src="https://github.com/lucianoAvinas/lapgm/raw/main/images/biased_data.png">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="350" src="https://github.com/lucianoAvinas/lapgm/raw/main/images/normalized_data.png">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

# Usage

Below we list some common commands and procedures to help with the debiasing and normalization of MR images.

**lapgm**'s GPU compute can be turned on and off globally with the 'use_gpu' command. Returned values will be loaded off of the GPU.
```python
import lapgm

# Set compute to GPU
lapgm.use_gpu(True)

# Set compute back to CPU
lapgm.use_gpu(False)
```

Before running for debiasing, LapGM's hyperparameters must be specified. 
```python
# takes in optional downscale_factor and other saving meta-settings
debias_obj = lapgm.LapGM()

# required: inverse penalty strength 'tau' and number of class 'n_classes'
debias_obj.set_hyperparameters(tau=tau, n_classes=n_classes)
```

The cylindrical weighting scheme used by [Vinas et al.](https://arxiv.org) is provided as:
```python
# required: penalty relaxation alpha which goes as r^(-alpha)
# optional: center, semi-major axes, and symmetries of the cylindrical ellipse
debias_obj.specify_cylindrical_decay(alpha=alpha)
```

Debiasing can be run as:
```python
# before running, disambiguate channeled data from spatial data with 'to_sequence array'
im_arr = lapgm.to_sequence_array([im_seq1, im_seq2])

# retrive estimated parameters
params = debias_obj.estimate_parameters(im_arr)

# get debiased result
db_arr = lapgm.debias(im_arr, params)
```

Normalization can be run using the same debiased sequence data and estimated parameters from above:
```python
# specify a target intensity to achieve
TRGT = 1000.

# normalize debiased array from before using estimated parameters and a target intensity
norm_arr = lapgm.normalize(db_arr, params, TRGT)
```

# References
1. Chris A. Cocosco et al. “BrainWeb: Online Interface to a 3D MRI Simulated Brain Database”. In: NeuroImage 5 (1997), p. 425.