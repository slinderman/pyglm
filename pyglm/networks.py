"""
Define some "networks" -- hierarchical prior distributions on the
weights of a set of regression objects.
"""
import abc
import numpy as np

from pybasicbayes.abstractions import GibbsSampling
from pybasicbayes.distributions import Gaussian

from pyglm.utils.utils import expand_scalar, expand_cov

class _NetworkModel(GibbsSampling):
    def __init__(self, N, B, **kwargs):
        """
        Only extra requirement is that we explicitly tell it the
        number of nodes and the dimensionality of the weights in the constructor.

        :param N: Number of nodes
        :param B: Dimensionality of the weights
        """
        self.N, self.B = N, B

    @abc.abstractmethod
    def resample(self,data=[]):
        """
        Every network mixin's resample method should call its parent in its
        first line. That way we can ensure that this base method is called
        first, and that each mixin is resampled appropriately.

        :param data: an adjacency matrix and a weight matrix
            A, W = data
            A in [0,1]^{N x N} where rows are incoming and columns are outgoing nodes
            W in [0,1]^{N x N x B} where rows are incoming and columns are outgoing nodes
        """
        assert isinstance(data, tuple)
        A, W = data
        N, B = self.N, self.B
        assert A.shape == (N, N)
        assert A.dtype == bool
        assert W.shape == (N, N, B)

    @abc.abstractproperty
    def mu_W(self):
        """
        NxNxB array of mean weights
        """
        raise NotImplementedError

    @abc.abstractproperty
    def sigma_W(self):
        """
        NxNxBxB array with conditional covariances of each weight
        """
        raise NotImplementedError

    @abc.abstractproperty
    def rho(self):
        """
        Connection probability
        :return: NxN matrix with values in [0,1]
        """
        pass

    ## TODO: Add properties for info form weight parameters

    def log_likelihood(self, x):
        # TODO
        return 0

    def rvs(self,size=[]):
        # TODO
        return None


class _IndependentGaussianMixin(_NetworkModel):
    """
    Each weight is an independent Bernoulli.
    Special case the self-connections.
    """
    def __init__(self, N, B,
                 mu_0=0.0, sigma_0=1.0, kappa_0=1.0, nu_0=3.0,
                 is_diagonal_weight_special=True,
                 **kwargs):
        super(_IndependentGaussianMixin, self).__init__(N, B)

        mu_0 = expand_scalar(mu_0, (B,))
        sigma_0 = expand_cov(sigma_0, (B,B))
        self._gaussian = Gaussian(mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, nu_0=nu_0)

        self.is_diagonal_weight_special = is_diagonal_weight_special
        if is_diagonal_weight_special:
            self._self_gaussian = \
                Gaussian(mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, nu_0=nu_0)

    @property
    def mu_W(self):
        N, B = self.N, self.B
        mu = np.zeros((N, N, B))
        if self.is_diagonal_weight_special:
            # Set off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            mu[mask] = self._gaussian.mu

            # set diagonal weights
            mask = np.eye(N).astype(bool)
            mu[mask] = self._self_gaussian.mu

        else:
            mu = np.tile(self._gaussian.mu[None,None,:], (N, N, 1))
        return mu

    @property
    def sigma_W(self):
        N, B = self.N, self.B
        if self.is_diagonal_weight_special:
            sigma = np.zeros((N, N, B, B))
            # Set off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            sigma[mask] = self._gaussian.sigma

            # set diagonal weights
            mask = np.eye(N).astype(bool)
            sigma[mask] = self._self_gaussian.sigma

        else:
            sigma = np.tile(self._gaussian.mu[None, None, :, :], (N, N, 1, 1))
        return sigma

    def resample(self, data=[]):
        super(_IndependentGaussianMixin, self).resample(data)
        A, W = data
        N, B = self.N, self.B
        if self.is_diagonal_weight_special:
            # Resample prior for off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            mask = mask & A
            self._gaussian.resample(W[mask])

            # Resample prior for diagonal weights
            mask = np.eye(N).astype(bool) & A
            self._self_gaussian.resample(W[mask])

        else:
            # Resample prior for all weights
            self._gaussian.resample(W[A])

class _FixedWeightsMixin(_NetworkModel):
    def __init__(self, N, B,
                 mu=0.0, sigma=1.0,
                 mu_self=None, sigma_self=None,
                 **kwargs):
        super(_FixedWeightsMixin, self).__init__(N, B)
        self._mu = expand_scalar(mu, (N, N, B))
        self._sigma = expand_cov(mu, (N, N, B, B))

        if (mu_self is not None) and (sigma_self is not None):
            self._mu[np.arange(N), np.arange(N), :] = expand_scalar(mu_self, (N, B))
            self._sigma[np.arange(N), np.arange(N), :] = expand_cov(sigma_self, (N, B, B))

    @property
    def mu_W(self):
        return self._mu

    @property
    def sigma_W(self):
        return self._sigma

    def resample(self,data=[]):
        super(_FixedWeightsMixin, self).resample(data)


class _IndependentBernoulliMixin(_NetworkModel):

    def __init__(self, N, B,
                 a_0=1.0, b_0=1.0,
                 is_diagonal_conn_special=True,
                 **kwargs):
        super(_IndependentBernoulliMixin, self).__init__(N, B)
        raise NotImplementedError("TODO: Implement the BetaBernoulli class")

        assert np.isscalar(a_0)
        assert np.isscalar(b_0)
        self._betabernoulli = BetaBernoulli(a_0, b_0)

        self.is_diagonal_conn_special = is_diagonal_conn_special
        if is_diagonal_conn_special:
            self._self_betabernoulli = BetaBernoulli(a_0, b_0)

    @property
    def rho(self):
        N, B = self.N, self.B
        rho = np.zeros((N, N))
        if self.is_diagonal_conn_special:
            # Set off-diagonal weights
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            rho[mask] = self._betabernoulli.rho

            # set diagonal weights
            mask = np.eye(N).astype(bool)
            rho[mask] = self._self_betabernoulli.rho

        else:
            rho = self._betabernoulli.rho * np.ones((N, N))
        return rho

    def resample(self, data=[]):
        super(_IndependentBernoulliMixin, self).resample(data)
        A, W = data
        N, B = self.N, self.B
        if self.is_diagonal_conn_special:
            # Resample prior for off-diagonal conns
            mask = np.ones((N, N), dtype=bool)
            mask[np.diag_indices(N)] = False
            self._betabernoulli.resample(A[mask])

            # Resample prior for off-diagonal conns
            mask = np.eye(N).astype(bool)
            self._self_betabernoulli.resample(A[mask])

        else:
            # Resample prior for all conns
            mask = np.ones((N, N), dtype=bool)
            self._betabernoulli.resample(A[mask])


class _FixedAdjacencyMixin(_NetworkModel):
    def __init__(self, N, B, rho=0.5, rho_self=None, **kwargs):
        super(_FixedAdjacencyMixin, self).__init__(N, B)
        self._rho = expand_scalar(rho, (N, N))
        if rho_self is not None:
            self._rho[np.diag_indices(N)] = rho_self

    @property
    def rho(self):
        return self._rho

    def resample(self,data=[]):
        super(_FixedAdjacencyMixin, self).resample(data)



class _DenseAdjacencyMixin(_NetworkModel):
    def __init__(self, N, B, **kwargs):
        super(_DenseAdjacencyMixin, self).__init__(N, B)
        self._rho = np.ones((N,N))

    @property
    def rho(self):
        return self._rho

    def resample(self,data=[]):
        super(_DenseAdjacencyMixin, self).resample(data)


### Define different combinations of network models
class FixedMeanDenseNetwork(_DenseAdjacencyMixin,
                            _FixedWeightsMixin):
    pass

class FixedMeanSparseNetwork(_FixedAdjacencyMixin,
                             _FixedWeightsMixin):
    pass

class NIWDenseNetwork(_DenseAdjacencyMixin,
                      _IndependentGaussianMixin):
    pass

class NIWFixedSparsityNetwork(_FixedAdjacencyMixin,
                              _IndependentGaussianMixin):
    pass
