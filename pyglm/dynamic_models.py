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

from pyglm.utils.utils import logistic

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
        self.omega = np.zeros((self.T, self.N))

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
    def log_likelihood_given_activation(self, psi):
        p   = logistic(psi)
        p   = np.clip(p, 1e-32, 1-1e-32)

        ll = (self.X * np.log(p) + (1-self.X) * np.log(1-p))
        return ll

    def rvs(self, psi):
        p = logistic(psi)
        return np.random.rand(*p.shape) < p



class NegativeBinomialObservations(_PGObservationsBase):
    # TODO: Implement a and b!
    pass
    def __init__(self, xi=10., **kwargs):
        super(NegativeBinomialObservations, self).__init__(**kwargs)
        assert xi > 0
        self.xi = xi

    def rvs(self, psi):
        p = logistic(psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        return np.random.negative_binomial(self.xi, 1-p)

    def log_likelihood_given_activation(self, psi):
        p   = logistic(psi)
        p   = np.clip(p, 1e-32, 1-1e-32)

        return self.log_normalizer(self.X, self.xi) \
               + self.X * np.log(p) \
               + self.xi * np.log(1-p)

    @staticmethod
    def log_normalizer(S, xi):
        return gammaln(S+xi) - gammaln(xi) - gammaln(S+1)

class PGEmissions(Distribution):
    """
    A base class for the emission matrix, C.
    """
    def __init__(self, D_out, D_in, C=None, sigmasq_C=1.):
        """
        :param D_out: Observation dimension
        :param D_in: Latent dimension
        :param C: Initial NxD emission matrix
        :param sigmasq_C: prior variance on C
        """
        self.D_out, self.D_in, self.sigmasq_C = D_out, D_in, sigmasq_C

        """self.mean_C  = np.zeros((self.D_out, self.D_in)) 
        self.sigma_Cns = np.zeros((self.D_out, self.D_in, self.D_in))
        for i in range(self.D_out):
            self.sigma_Cns[i, :, :] = sigmasq_C * np.eye(self.D_in)
        """
        
        if C:
            assert C.shape == (self.D_out, self.D_in)
            self.C = C
        else:
            self.C = np.sqrt(sigmasq_C) * np.random.rand(self.D_out, self.D_in)

    def resample(self, states_list):
        zs = [s.stateseq for s in states_list]
        kappas = [s.observations.kappa for s in states_list]
        omegas = [s.observations.omega for s in states_list]

        assert(len(kappas) = len(omegas))
        import pdb
        pdb.set_trace()

        
        z = np.hstack(zs)
        kappa = np.hstack(kappas)
        omega = np.hstack(omegas)

        sIinv = (1 / self.sigmasq_C) * np.eye((self.D_in, self.D_in))
        for n in xrange(self.D_out):
            # TODO: Resample C_{n,:} given z and omega[:,n]
            Omega = np.diag(1 / omega[:, n])
            
            zOz = np.dot(np.dot(z.T, Omega), z)
            sigmainv = sIinv + z0z
            self.C[n, :] = np.random.randn(
                np.dot(np.dot(kappa[:, T].T, z), sigmainv),
                np.linalg.inv(sigmainv)
            )
            
                    

    def rvs(self,size=[],x=None):
        assert x.ndim==2 and x.shape[1] == self.D_in
        psi = x.dot(self.C.T)
        return psi


class PGLDSStates(LDSStates):
    def __init__(self, model, *args, **kwargs):
        assert isinstance(model, _PGLDSBase)
        super(PGLDSStates, self).__init__(model, *args, **kwargs)

    def resample(self):

        # Have the observation object compute the conditional mean and covariance
        conditional_mean = self.data.conditional_mean()
        conditional_cov = self.data.conditional_cov(flat=True)

        ll, self.stateseq = filter_and_sample_diagonal(
            self.mu_init, self.sigma_init,
            self.A, self.sigma_states,
            self.C, conditional_cov, conditional_mean)

        assert np.all(np.isfinite(self.stateseq))

    def log_likelihood(self, M=100):
        """
        Compute conditional log likelihood by Monte Carlo sampling omega
        """
        # TODO: We can derive better ways of estimating the log likelihood
        lls = np.zeros(M)
        for m in xrange(M):
            # Sample latent states from the prior
            z = self.generate_states()
            psi = z.dot(self.C.T)
            lls[m] = self.data.log_likelihood_given_activation(psi)

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

    def __init__(self, observation_kwargs={}, **kwargs):
        super(_PGLDSBase, self).__init__(**kwargs)
        self._observation_kwargs.update(observation_kwargs)

    def add_data(self, data, **kwargs):
        assert isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == self.n
        obs = self._observation_class(data, **self._observation_kwargs)
        self.states_list.append(PGLDSStates(model=self, data=obs))

    def _generate_obs(self, s):
        if s.data is None:
            psi = self.emission_distn.rvs(x=s.stateseq)
            data = self._observation_class(psi=psi, **self._observation_kwargs)
            s.data = data
        else:
            # filling in missing data
            raise NotImplementedError

        return s.data


class BernoulliLDS(_PGLDSBase):
    _observation_class = BernoulliObservations
    _observation_kwargs = {}


class NegativeBinomialLDS(_PGLDSBase):
    _observation_class = NegativeBinomialObservations
    _observation_args = {"xi": 10.}
