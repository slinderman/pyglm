import abc
import numpy as np
from scipy.special import gammaln

# from hips.distributions.polya_gamma import polya_gamma
from pypolyagamma import pgdrawv, PyRNG

from pyglm.abstractions import Component
from pyglm.utils.utils import logistic

class _PolyaGammaAugmentedObservationsBase(Component):
    """
    Class to keep track of a set of spike count observations and the
    corresponding Polya-gamma auxiliary variables associated with them.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, population):
        self.population = population
        self.rng = PyRNG()

    @property
    def activation(self):
        return self.population.activation_model

    def augment_data(self, augmented_data):
        """
        Add a matrix of augmented counts
        :param augmented_data:
        :return:
        """
        S = augmented_data["S"]
        self.T = S.shape[0]
        self.N = S.shape[1]

        # Initialize auxiliary variables
        augmented_data["omega"] = np.empty((self.T, self.N))
        for n in xrange(self.N):
            tmp = np.empty(self.T)
            pgdrawv(np.ones(self.T, dtype=np.int32),
                    np.zeros(self.T),
                    tmp, self.rng)
            augmented_data["omega"][:,n] = tmp

        # Precompute kappa (assuming that it is constant given data)
        augmented_data["kappa"] = self.a(augmented_data) - self.b(augmented_data)/2.0

        # Initialize the mean field local variational parameters
        augmented_data["omega"] = np.empty((self.T, self.N))

    @abc.abstractmethod
    def a(self, augmented_data):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def b(self, augmented_data):
        """
        The exponent in the denominator of the logistic likelihood
            exp(\psi)^a / (1+exp(\psi)^b
        """
        raise NotImplementedError()

    def kappa(self, augmented_data):
        """
        Compute kappa = b-a/2
        :return:
        """
        # return self.a(augmented_data) - self.b(augmented_data)/2.0
        return augmented_data["kappa"]

    def omega(self, augmented_data):
        return augmented_data["omega"]

    @abc.abstractmethod
    def rvs(self, Psi):
        raise NotImplementedError()

    @abc.abstractmethod
    def expected_S(self, Psi):
        raise NotImplementedError()

    def resample(self, augmented_data):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        psi = self.activation.compute_psi(augmented_data)

        # Resample the auxiliary variables, omega, in Python
        # self.omega = polya_gamma(self.conditional_b.reshape(self.T),
        #                          self.psi.reshape(self.T),
        #                          200).reshape((self.T,))

        # Create a PyPolyaGamma object and resample with the C code
        # seed = np.random.randint(2**16)
        # ppg = PyPolyaGamma(seed, self.model.trunc)
        # ppg.draw_vec(self.conditional_b, self.psi, self.omega)

        # Resample with Jesse Windle's ported code
        b = self.b(augmented_data)
        for n in xrange(self.N):
            bn   = b[:,n].copy("C")
            psin = psi[:,n].copy("C")
            tmpn = np.empty(augmented_data["T"])
            pgdrawv(bn,
                    psin,
                    tmpn,
                    self.rng)
            augmented_data["omega"][:,n] = tmpn

    ### Mean field
    def meanfieldupdate(self, augmented_data):
        """
        Compute the expectation of omega under the variational posterior.
        This requires us to sample activations and perform a Monte Carlo
        integration.
        """
        Psis = self.activation.mf_sample_activation(augmented_data, N_samples=10)
        augmented_data["E_omega"] = self.b(augmented_data) / 2.0 \
                                    * (np.tanh(Psis/2.0) / (Psis)).mean(axis=0)

    def mf_expected_omega(self, augmented_data):
        # DEBUG
        # self.meanfieldupdate(augmented_data)
        return augmented_data["E_omega"]

    @abc.abstractmethod
    def expected_log_likelihood(self, augmented_data, expected_suff_stats):
        """
        Compute the expected log likelihood with expected parameters x
        """
        raise NotImplementedError()

    def get_vlb(self, augmented_data):
        # 1. E[ \ln p(s | \psi) ]
        # Compute this with Monte Carlo integration over \psi
        Psis = self.activation.mf_sample_activation(augmented_data, N_samples=10)
        ps = logistic(Psis)
        E_lnp = np.log(ps).mean(axis=0)
        E_ln_notp = np.log(1-ps).mean(axis=0)

        vlb = self.expected_log_likelihood(augmented_data,
                                           (E_lnp, E_ln_notp)).sum()
        return vlb

    def resample_from_mf(self, augmented_data):
        # This is a no-op for the observation model
        pass

    ### SVI
    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        """
        The observations only have global parameters, so the SVI
        step is the same as a standard mean field update.
        """
        self.meanfieldupdate(augmented_data)


class BernoulliObservations(_PolyaGammaAugmentedObservationsBase):
    def log_likelihood(self, augmented_data):
        S   = augmented_data["S"]
        Psi = self.activation.compute_psi(augmented_data)
        p   = logistic(Psi)
        p   = np.clip(p, 1e-32, 1-1e-32)

        ll = (S * np.log(p) + (1-S) * np.log(1-p))
        return ll

    def a(self, augmented_data):
        return augmented_data["S"]

    def b(self, augmented_data):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        return np.ones_like(augmented_data["S"])

    def rvs(self, Psi):
        p = logistic(Psi)
        return np.random.rand(*p.shape) < p

    def expected_S(self, Psi):
        p = logistic(Psi)
        return p

    def expected_log_likelihood(self, augmented_data, expected_suff_stats):
        """
        Compute the expected log likelihood with expected parameters x
        """
        S = augmented_data["S"]
        E_ln_p, E_ln_notp = expected_suff_stats
        return S * E_ln_p + (1-S) * E_ln_notp


class NegativeBinomialObservations(_PolyaGammaAugmentedObservationsBase):
    def __init__(self, population, xi=1.0):
        super(NegativeBinomialObservations, self).__init__(population)

        assert xi > 0, "Xi must greater than 0 for negative binomial NB(xi, p)"
        self.xi = xi

    def log_likelihood(self, augmented_data):
        S = augmented_data["S"]
        Psi = self.activation.compute_psi(augmented_data)
        p   = logistic(Psi)
        p   = np.clip(p, 1e-32, 1-1e-32)

        return self.log_normalizer(S) + S * np.log(p) + self.xi * np.log(1-p)

    def log_normalizer(self, S):
        return gammaln(S+self.xi) - gammaln(self.xi) - gammaln(S+1)

    def a(self, augmented_data):
        return augmented_data["S"]

    def b(self, augmented_data):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        return augmented_data["S"] + self.xi

    def rvs(self, Psi):
        p = logistic(Psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        return np.random.negative_binomial(self.xi, 1-p)

    def expected_S(self, Psi):
        p = logistic(Psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        return self.xi * p / (1-p)


    def expected_log_likelihood(self, augmented_data, expected_suff_stats):
        """
        Compute the expected log likelihood with expected parameters x
        """
        S = augmented_data["S"]
        E_ln_p, E_ln_notp = expected_suff_stats
        return self.log_normalizer(S) + S * E_ln_p + self.xi * E_ln_notp