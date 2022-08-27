import cupy as cp
import numpy as np
import scipy.linalg

from .typing_details import Array

# Original scipy.stats._multivariate.py author: Joris Vankerschaver 2013
# Rewritten using cupy functions


class multivariate_normal:
    @staticmethod
    def pdf(x: Array[float, ('M', 'N')], mean: Array[float, ('M', 'K')],
            cov: Array[float, ('M', 'K', 'K')], allow_singular: bool = False):
        """Calculates probability density value for a specified multivariate normal.

        Args:
            x: Multivariate values to calculate probability density on.
            mean: Mean of multivariate normal.
            cov: Covariance of multivariate normal.
            allow_singular: Throws exception if true and cov singular.

        Returns array of length 'N' of probability density values.
        """
        dim = mean.shape[0]

        s, u = scipy.linalg.eigh(cp.asnumpy(cov), lower=True, check_finite=True)

        t = s.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        eps = factor[t] * np.finfo(t).eps * np.max(abs(s))

        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')

        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            raise np.linalg.LinAlgError('singular matrix')

        s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
        U = cp.asarray(np.multiply(u, np.sqrt(s_pinv)))

        rank = len(d)
        log_pdet = np.sum(np.log(d))

        dev = x - mean
        maha = cp.sum(cp.square(cp.dot(dev, U)), axis=-1)
        out = cp.exp(-0.5 * (rank * np.log(2 * np.pi) + log_pdet + maha))

        out = out.squeeze()
        if out.ndim == 0:
            out = out[()]

        return out
