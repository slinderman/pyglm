"""
Some simple models with multinomial observations and Gaussian priors.
"""
from __future__ import print_function
import abc
import numpy as np

from scipy.misc import logsumexp

from pybasicbayes.abstractions import Distribution
from pylds.states import LDSStates
from pylds.lds_messages_interface import filter_and_sample_diagonal, kalman_filter
from pylds.models import NonstationaryLDS

import pypolyagamma as ppg

class _PGObservationsBase(object):
    """
    Base class for Polya-gamma observations
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, X):
        """
        :param X: TxN matrix of observations
        """
        assert X.ndim == 2
        self.X = X
        self.T, self.N = X.shape

        # Initialize Polya-gamma samplers
        num_threads = ppg.get_omp_num_threads()
        seeds = np.random.randint(2**16, size=num_threads)
        self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

        # Initialize auxiliary variables, omega
        self.omega = np.zeros((self.T, self.N))

    @abc.abstractproperty
    def a(self):
        # Distribution specific exponent a(x)
        raise NotImplementedError()

    @abc.abstractproperty
    def b(self):
        # Distribution specific exponent a(x)
        raise NotImplementedError()

    @property
    def kappa(self):
        # TODO: Cache this?
        return self.a - self.b / 2.0

    def conditional_mean(self):
        """
        Compute the conditional mean \psi given \omega
        :param augmented_data:
        :return:
        """
        return self.kappa / self.omega

    def conditional_prec(self, flat=False):
        """
        Compute the conditional mean \psi given \omega
        """
        O, T, N = self.omega, self.T, self.N

        if flat:
            prec = O
        else:
            prec = np.zeros((T, N, N))
            for t in xrange(T):
                prec[t,:,:] = np.diag(O[t,:])

        return prec

    def conditional_cov(self, flat=False):
        O, T, N = self.omega, self.T, self.N

        if flat:
            cov = 1./O
        else:
            cov = np.zeros((T, N, N))
            for t in xrange(T):
                cov[t,:,:] = np.diag(1./O[t,:])

        return cov


class BernoulliObservations(_PGObservationsBase):
    # TODO: Implement a and b!
    pass


class NegativeBinomialObservations(_PGObservationsBase):
    # TODO: Implement a and b!
    pass


class PGEmissions(Distribution):
    """
    A base class for the emission matrix, C.
    """
    def __init__(self, model, C=None, sigmasq_C=1.):
        """
        :param N: Observation dimension
        :param D: Latent dimension
        :param C: Initial NxD emission matrix
        :param sigma_C: prior variance of emission matrix entries
        """
        self.model = model
        self.D_out, self.D_in, self.sigmasq_C = model.p, model.n, sigmasq_C

        if C:
            assert C.shape == (self.D_out, self.D_in)
            self.C = C
        else:
            self.C = np.sqrt(sigmasq_C) * np.random.rand(self.D_out, self.D_in)

    def resample(self, states_list):
        zs = [s.stateseq for s in states_list]
        omegas = [s.observations.omega for s in states_list]

        for n in xrange(self.D_out):
            # TODO: Resample C_{n,:} given z and omega[:,n]
            pass


class PGLDSStates(LDSStates):
    def __init__(self, model, *args, **kwargs):
        assert isinstance(model, _PGLDSBase)
        super(PGLDSStates, self).__init__(model, *args, **kwargs)

    @property
    def observations(self):
        return self.model.observations

    def resample(self):

        # Have the observation object ompute the conditional mean and covariance
        conditional_mean = self.observations.conditional_mean(self.data)
        conditional_cov = self.observations.conditional_cov(self.data, flat=True)

        ll, self.stateseq = filter_and_sample_diagonal(
            self.mu_init, self.sigma_init,
            self.A, self.sigma_states,
            self.C, conditional_cov, conditional_mean)

        assert np.all(np.isfinite(self.stateseq))

    def log_likelihood(self):
        """
        Compute conditional log likelihood given omega
        """
        # Have the observation object ompute the conditional mean and covariance
        conditional_mean = self.observations.conditional_mean(self.data)
        conditional_cov = self.observations.conditional_cov(self.data, flat=True)

        assert conditional_mean.shape == (self.T, self.p)
        assert conditional_cov.shape == (self.T, self.p, self.p)

        normalizer, _, _ = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.sigma_states,
            self.C, conditional_cov, conditional_mean)
        return normalizer


class _PGLDSBase(NonstationaryLDS):
    _observation_class = _PGObservationsBase


    def add_data(self, data, **kwargs):
        assert isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == self.n
        obs = self._observation_class(data)
        self.states_list.append(PGLDSStates(model=self, data=obs))

    def heldout_log_likelihood(self, X, M=100):
        return self._mc_heldout_log_likelihood(X, M)

    def _mc_heldout_log_likelihood(self, X, M=100):
        raise NotImplementedError()
        # Estimate the held out likelihood using Monte Carlo
        T, K = X.shape
        assert K == self.K

        lls = np.zeros(M)
        for m in xrange(M):
            # Sample latent states from the prior
            states = self.generate(T=T, keep=False)
            data["x"] = X
            lls[m] = self.emission_distn.log_likelihood(data)

        # Compute the average
        hll = logsumexp(lls) - np.log(M)

        # Use bootstrap to compute error bars
        samples = np.random.choice(lls, size=(100, M), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(M)
        std_hll = hll_samples.std()

        return hll, std_hll

    def _generate_obs(self, s):
        raise NotImplementedError
        if s.data is None:
            # TODO: Compute psi from z and C
            # TODO: Sample rvs using self.observation
            s.data = self.emission_distn.rvs(x=s.stateseq,return_xy=False)
        else:
            # filling in missing data
            raise NotImplementedError
        return s.data


class BernoulliLDS(_PGLDSBase):
    _observation_class = BernoulliObservations


class NegativeBinomialLDS(_PGLDSBase):
    _observation_class = NegativeBinomialObservations


