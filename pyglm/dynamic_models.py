"""
Some simple models with multinomial observations and Gaussian priors.
"""
from __future__ import print_function
import abc
import numpy as np

from scipy.misc import logsumexp
from scipy.special import gammaln

from pybasicbayes.abstractions import Distribution
from pylds.states import LDSStates
from pylds.lds_messages_interface import filter_and_sample_diagonal, kalman_filter
from pylds.models import NonstationaryLDS

import pypolyagamma as ppg

from pyglm.utils.utils import logistic, logit, sample_gaussian

class _PGObservationsBase(object):
    """
    Base class for Polya-gamma observations
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, X=None, psi=None):
        """
        :param X: TxN matrix of observations
        """
        assert X is not None or psi is not None
        if psi is not None and X is None:
            X = self.rvs(psi)

        assert X.ndim == 2
        self.X = X
        self.T, self.N = X.shape

        # Initialize Polya-gamma samplers
        num_threads = ppg.get_omp_num_threads()
        seeds = np.random.randint(2**16, size=num_threads)
        self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

        # Initialize auxiliary variables, omega
        # Get an approximate value of psi
        psi0 = self.invert_rate(self.empirical_rate())
        self.omega = np.ones((self.T, self.N))
        ppg.pgdrawvpar(self.ppgs, self.b.ravel(), psi0.ravel(), self.omega.ravel())

    @abc.abstractproperty
    def a(self):
        # Distribution specific exponent a(x)
        raise NotImplementedError()

    @abc.abstractproperty
    def b(self):
        # Distribution specific exponent a(x)
        raise NotImplementedError()

    @abc.abstractmethod
    def log_likelihood_given_activation(self, psi):
        raise NotImplementedError()

    @abc.abstractmethod
    def rvs(self, psi):
        raise NotImplementedError

    @abc.abstractmethod
    def rate(self, psi):
        raise NotImplementedError

    @abc.abstractmethod
    def invert_rate(self, rate):
        raise NotImplementedError

    def empirical_rate(self, sigma=3.0):
        """
        Smooth X to get an empirical rate
        """
        from scipy.ndimage.filters import gaussian_filter1d
        return 0.001 + gaussian_filter1d(self.X.astype(np.float), sigma, axis=0)

    @property
    def kappa(self):
        # TODO: Cache this?
        return self.a - self.b / 2.0

    def resample(self, psi):
        ppg.pgdrawvpar(self.ppgs, self.b.ravel(), psi.ravel(),
                       self.omega.ravel())

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
    @property
    def a(self):
        return self.X

    @property
    def b(self):
        return np.ones_like(self.X)

    def rate(self, psi):
        return logistic(psi)

    def invert_rate(self, rate):
        return logit(rate)

    def log_likelihood_given_activation(self, psi):
        p   = logistic(psi)
        p   = np.clip(p, 1e-16, 1-1e-16)

        ll = (self.X * np.log(p) + (1-self.X) * np.log(1-p))
        return ll

    def rvs(self, psi):
        p = logistic(psi)
        return (np.random.rand(*p.shape) < p).astype(np.float)



class NegativeBinomialObservations(_PGObservationsBase):
    def __init__(self, xi=10., **kwargs):
        assert xi > 0
        self.xi = xi

        super(NegativeBinomialObservations, self).__init__(**kwargs)

    @property
    def a(self):
        return self.X

    @property
    def b(self):
        return self.X + self.xi

    def rate(self, psi):
        return self.xi * np.exp(psi)

    def invert_rate(self, rate):
        # Mean is r * exp(logit(p))
        # = r * exp(logit(logistic(psi)))
        # = r * exp(psi)
        # psi = log(mean / r)
        return np.log(rate / self.xi)

    def rvs(self, psi):
        p = logistic(psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        return np.random.negative_binomial(self.xi, 1-p).astype(np.float)

    def log_likelihood_given_activation(self, psi):
        p   = logistic(psi)
        p   = np.clip(p, 1e-32, 1-1e-32)

        return self.log_normalizer(self.X, self.xi) \
               + self.X * np.log(p) \
               + self.xi * np.log(1-p)

    @staticmethod
    def log_normalizer(S, xi):
        return gammaln(S+xi) - gammaln(xi) - gammaln(S+1)


class ApproxPoissonObservations(NegativeBinomialObservations):
    """
    Approximate Poisson(e^psi) with NB(xi, \sigma(psi - log xi)) for large xi
    """
    def __init__(self, xi=500.0, **kwargs):
        super(ApproxPoissonObservations, self).__init__(xi=xi, **kwargs)

    def rate(self, psi):
        return np.exp(psi)

    def invert_rate(self, rate):
        # Mean is r * exp(logit(p))
        # = r * exp(logit(logistic(psi)))
        # = r * exp(psi)
        # psi = log(mean / r)
        return np.log(rate / self.xi)

    def resample(self, psi):
        ppg.pgdrawvpar(self.ppgs, self.b.ravel(),
                       (psi - np.log(self.xi)).ravel(),
                       self.omega.ravel())
        assert np.all(np.isfinite(self.omega))

    def conditional_mean(self):
        """
        Compute the conditional mean \psi given \omega
        :param augmented_data:
        :return:
        """
        return self.kappa / self.omega + np.log(self.xi)

    def rvs(self, psi):
        p = logistic(psi - np.log(self.xi))
        p = np.clip(p, 1e-32, 1-1e-32)
        return np.random.negative_binomial(self.xi, 1-p).astype(np.float)

    def log_likelihood_given_activation(self, psi):
        p   = logistic(psi - np.log(self.xi))
        p   = np.clip(p, 1e-32, 1-1e-32)

        return self.log_normalizer(self.X, self.xi) \
               + self.X * np.log(p) \
               + self.xi * np.log(1-p)


class TruePoissonObservations(_PGObservationsBase):
    """
    Model for simulating truly Poisson observations
    (even though we can't do  exact inference in this model
    with the PG trick.)
    """
    def __init__(self, X=None, psi=None):
        """
        :param X: TxN matrix of observations
        """
        assert X is not None or psi is not None
        if psi is not None and X is None:
            X = self.rvs(psi)

        assert X.ndim == 2
        self.X = X
        self.T, self.N = X.shape

    @property
    def a(self):
        # Distribution specific exponent a(x)
        raise NotImplementedError()

    @property
    def b(self):
        # Distribution specific exponent a(x)
        raise NotImplementedError()

    def rate(self, psi):
        return np.exp(psi)

    def invert_rate(self, rate):
        return np.log(rate)

    def log_likelihood_given_activation(self, psi):
        """
        -lmbda + np.log(lmbda) * x - gammaln(x+1)
        """
        lmbda = np.exp(psi)
        ll = -lmbda + self.X * np.log(lmbda) - gammaln(self.X+1)
        return ll

    def rvs(self, psi):
        lmbda = np.exp(psi)
        return np.random.poisson(lmbda)


class PGEmissions(Distribution):
    """
    A base class for the emission matrix, C.
    """
    def __init__(self, D_out, D_in, C=None, sigmasq_C=1.,
                 b=None, mu_b=0., sigmasq_b=1.):
        """
        :param D_out: Observation dimension
        :param D_in: Latent dimension
        :param C: Initial NxD emission matrix
        :param sigmasq_C: prior variance on C
        :param b: Initial Nx1 emission matrix
        :param sigmasq_b: prior variance on b
        """
        self.D_out, self.D_in, self.sigmasq_C, self.mu_b, self.sigmasq_b = \
            D_out, D_in, sigmasq_C, mu_b, sigmasq_b

        if C is not None:
            assert C.shape == (self.D_out, self.D_in)
            self.C = C
        else:
            self.C = np.sqrt(sigmasq_C) * np.random.rand(self.D_out, self.D_in)

        if b is not None:
            assert b.shape == (self.D_out, 1)
            self.b = b
        else:
            self.b = np.sqrt(sigmasq_b) * np.random.rand(self.D_out, 1)

    def resample(self, states_list):
        D = self.D_in
        for n in xrange(self.D_out):
            # Resample C_{n,:} given z, omega[:,n], and kappa[:,n]
            prior_h = np.zeros(D + 1)
            prior_h[D] = self.mu_b / self.sigmasq_b
            prior_J = 1./self.sigmasq_C * np.eye(D + 1)
            prior_J[D, D] = 1. / self.sigmasq_b

            lkhd_h = np.zeros(D + 1)
            lkhd_J = np.zeros((D + 1, D + 1))

            for states in states_list:
                z = states.stateseq
                # TODO: figure out how to do this more nicely later
                z = np.hstack((z, np.ones((z.shape[0], 1))))
                kappa = states.data.conditional_mean() / states.data.conditional_cov(flat=True)
                omega = states.data.conditional_prec(flat=True)

                # J += z.T.dot(diag(omega_n)).dot(z)
                lkhd_J += (z * omega[:,n][:,None]).T.dot(z)
                lkhd_h += kappa[:,n].T.dot(z)

            post_h = prior_h + lkhd_h
            post_J = prior_J + lkhd_J

            joint_sample = sample_gaussian(J=post_J, h=post_h)
            self.C[n,:]  = joint_sample[:D]
            self.b[n]    = joint_sample[D]

    def log_likelihood(self, C, b):
        # TODO: Normalize
        C_ll = -0.5 * (C**2 / self.sigmasq_C).sum()
        b_ll = -0.5 * (b ** 2  / self.sigmasq_b).sum()
        return C_ll + b_ll

    def rvs(self,size=[],x=None):
        assert x.ndim==2 and x.shape[1] == self.D_in
        psi = x.dot(self.C.T) + self.b.T
        
        # for n in xrange(self.D_out):
        #     psi[:,n] += self.b[n]
        
        return psi


class PGLDSStates(LDSStates):
    def __init__(self, model, data):
        assert isinstance(model, _PGLDSBase)
        assert isinstance(data, _PGObservationsBase)
        T = data.X.shape[0]
        super(PGLDSStates, self).__init__(model, data=data, T=T)

    @property
    def b(self):
        return self.emission_distn.b

    @property
    def psi(self):
        return self.stateseq.dot(self.C.T) + self.b.T

    @property
    def rate(self):
        return self.data.rate(self.psi)

    def resample(self):
        self.resample_states()
        self.resample_auxiliary_variables()

    def resample_states(self):
        # Have the observation object compute the conditional mean and covariance
        conditional_mean = self.data.conditional_mean() - self.b.T
        conditional_cov = self.data.conditional_cov(flat=True)

        ll, self.stateseq = filter_and_sample_diagonal(
            self.mu_init, self.sigma_init,
            self.A, self.sigma_states,
            self.C, conditional_cov, conditional_mean)

        assert np.all(np.isfinite(self.stateseq))

    def resample_auxiliary_variables(self):
        self.data.resample(self.psi)

    def log_likelihood(self):
        return self.data.log_likelihood_given_activation(self.psi).sum()

    def heldout_log_likelihood(self, M=100):
        """
        Compute conditional log likelihood by Monte Carlo sampling omega
        """
        # TODO: We can derive better ways of estimating the log likelihood
        lls = np.zeros(M)
        for m in xrange(M):
            # Sample latent states from the prior
            z = self.generate_states()
            psi = z.dot(self.C.T) + self.b.T
            lls[m] = self.data.log_likelihood_given_activation(psi).sum()

        # Compute the average
        hll = logsumexp(lls) - np.log(M)

        # Use bootstrap to compute error bars
        samples = np.random.choice(lls, size=(100, M), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(M)
        std_hll = hll_samples.std()

        return hll, std_hll


class _PGLDSBase(NonstationaryLDS):
    _observation_class = _PGObservationsBase
    _observation_kwargs = {}

    def __init__(self, init_dynamics_distn, dynamics_distn, emission_distn, observation_kwargs={}):
        assert isinstance(emission_distn, PGEmissions)
        super(_PGLDSBase, self).__init__(init_dynamics_distn=init_dynamics_distn,
                                         dynamics_distn=dynamics_distn,
                                         emission_distn=emission_distn)
        self._observation_kwargs.update(observation_kwargs)

    @property
    def C(self):
        return self.emission_distn.C

    def add_data(self, data, **kwargs):
        assert isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == self.p
        obs = self._observation_class(X=data, **self._observation_kwargs)
        states = PGLDSStates(model=self, data=obs)
        # Resample the stateseq a few times
        [states.resample_states() for _ in xrange(10)]
        self.states_list.append(states)


    def _generate_obs(self, s):
        if s.data is None:
            psi = self.emission_distn.rvs(x=s.stateseq)
            data = self._observation_class(psi=psi, **self._observation_kwargs)
            s.data = data
        else:
            # filling in missing data
            raise NotImplementedError

        return s.data

    def resample_parameters(self):
        self.resample_dynamics_distn()
        self.resample_emission_distn()
        # pass

    def resample_emission_distn(self):
        self.emission_distn.resample(self.states_list)


class BernoulliLDS(_PGLDSBase):
    _observation_class = BernoulliObservations
    _observation_kwargs = {}


class NegativeBinomialLDS(_PGLDSBase):
    _observation_class = NegativeBinomialObservations
    _observation_args = {"xi": 1.}

class PoissonLDS(_PGLDSBase):
    _observation_class = TruePoissonObservations
    _observation_args = {}

    def resample_model(self):
        raise NotImplementedError()


class ApproxPoissonLDS(_PGLDSBase):
    _observation_class = ApproxPoissonObservations
    _observation_args = {}