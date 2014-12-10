import numpy as np

from pyglm.deps.pybasicbayes.abstractions import GibbsSampling
from pypolyagamma import pgdrawv, PyRNG

class _PolyaGammaAugmentedCountsBase(GibbsSampling):
    """
    Class to keep track of a set of counts and the corresponding Polya-gamma
    auxiliary variables associated with them.
    """
    def __init__(self, X, counts, nbmodel):
        assert counts.ndim == 1
        self.counts = counts.astype(np.int32)
        self.T = counts.shape[0]

        # assert X.ndim == 2 and X.shape[0] == self.T
        self.X = X

        # Keep this pointer to the model
        self.model = nbmodel

        # Initialize auxiliary variables
        self.omega = np.ones(self.T)
        self.psi = self.model.mean_activation(X)

    def log_likelihood(self, x):
        return 0

    def rvs(self, size=[]):
        return None

class AugmentedNegativeBinomialCounts(_PolyaGammaAugmentedCountsBase):

    def resample(self, data=None, stats=None):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        # Create a PyPolyaGamma object
        seed = np.random.randint(2**16)
        # ppg = PyPolyaGamma(seed, self.model.trunc)
        rng = PyRNG()

        xi = np.int32(self.model.xi)
        mu = self.model.mean_activation(self.X)
        sigma = self.model.sigma

        sigma = np.asscalar(sigma)
        # Resample the auxiliary variables, omega
        # self.omega = polya_gamma(self.counts.reshape(self.T)+xi,
        #                          self.psi.reshape(self.T),
        #                          self.model.trunc).reshape((self.T,))

        # Resample with the C code
        # ppg.draw_vec(self.counts+xi, self.psi, self.omega)
        pgdrawv(self.counts+xi, self.psi, self.omega, rng)

        # Resample the rates, psi given omega and the regression parameters
        sig_post = 1.0 / (1.0/sigma + self.omega)
        mu_post = sig_post * ((self.counts-xi)/2.0 + mu / sigma)
        self.psi = mu_post + np.sqrt(sig_post) * np.random.normal(size=(self.T,))



# We can also do logistic regression as a special case!
class AugmentedBernoulliCounts(_PolyaGammaAugmentedCountsBase):
    def resample(self, data=None, stats=None):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        # Create a PyPolyaGamma object
        seed = np.random.randint(2**16)
        # ppg = PyPolyaGamma(seed, self.model.trunc)
        rng = PyRNG()

        # Resample with the C code
        # ppg.draw_vec(np.ones(self.T, dtype=np.int32), self.psi, self.omega)
        pgdrawv(np.ones(self.T, dtype=np.int32), self.psi, self.omega, rng)

        # Resample the rates, psi given omega and the regression parameters
        mu_prior = self.model.mean_activation(self.X)
        sigma_prior = self.model.sigma
        sigma_prior = np.asscalar(sigma_prior)

        sig_post = 1.0 / (1.0/sigma_prior + self.omega)
        mu_post = sig_post * (self.counts-0.5 + mu_prior / sigma_prior)
        self.psi = mu_post + np.sqrt(sig_post) * np.random.normal(size=(self.T,))

