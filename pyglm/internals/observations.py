import abc

import numpy as np
from scipy.special import gammaln

# from hips.distributions.polya_gamma import polya_gamma

from pyglm.deps.pybasicbayes.abstractions import GibbsSampling, MeanField
from pyglm.internals.distributions import ScalarGaussian
from pypolyagamma import pgdrawv, PyRNG

from pyglm.utils.utils import logistic


class _PolyaGammaAugmentedCountsBase(GibbsSampling, MeanField):
    """
    Class to keep track of a set of counts and the corresponding Polya-gamma
    auxiliary variables associated with them.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, counts, neuron):
        assert counts.ndim == 1
        self.counts = counts.astype(np.int32)
        self.T = counts.shape[0]

        # assert X.ndim == 2 and X.shape[0] == self.T
        self.X = X

        # Keep this pointer to the model
        self.neuron = neuron

        # Initialize auxiliary variables
        # self.omega = polya_gamma(np.ones(self.T),
        #                          self.psi.reshape(self.T),
        #                          200).reshape((self.T,))
        self.omega = np.ones(self.T)
        rng = PyRNG()
        pgdrawv(np.ones(self.T, dtype=np.int32), np.zeros(self.T), self.omega, rng)

    def log_likelihood(self, x):
        return 0

    def rvs(self, size=[]):
        return None

    @abc.abstractproperty
    def a(self):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def b(self):
        """
        The exponent in the denominator of the logistic likelihood
            exp(\psi)^a / (1+exp(\psi)^b
        """
        raise NotImplementedError()

    @property
    def kappa(self):
        """
        Compute kappa = b-a/2
        :return:
        """
        return self.a - self.b/2.0

    @property
    def psi(self):
        """
        The second parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi).
        By default, this is the model's mean activation.
        """
        return self.neuron.mean_activation(self.X)

    ### Gibbs Sampling
    def resample(self, data=None, stats=None):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        # Resample the auxiliary variables, omega, in Python
        # self.omega = polya_gamma(self.conditional_b.reshape(self.T),
        #                          self.psi.reshape(self.T),
        #                          200).reshape((self.T,))

        # Create a PyPolyaGamma object and resample with the C code
        # seed = np.random.randint(2**16)
        # ppg = PyPolyaGamma(seed, self.model.trunc)
        # ppg.draw_vec(self.conditional_b, self.psi, self.omega)

        # Resample with Jesse Windle's ported code
        rng = PyRNG()
        pgdrawv(self.b, self.psi, self.omega, rng)

    ### Mean Field
    def meanfieldupdate(self,data,weights):
        """
        Nothing to do since we do not keep variational parameters for q(omega).
        """
        pass

    def mf_expected_psi(self):
        """
        The second parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi).
        By default, this is the model's mean activation under the mean field
        approximation.
        """
        return self.neuron.mf_mean_activation(self.X)

    def mf_covariance_psi(self):
        """

        """
        return self.neuron.mf_covariance_activation(self.X)

    def mf_sample_psis(self, N_psis=100):
        """
        Sample psis from variational distribution
        """
        # TODO "Have the neuron sample weights and compute psi"
        psis = np.random.multivariate_normal(self.mf_expected_psi(),
                                             self.mf_covariance_psi(),
                                             size=N_psis)
        return psis

    def expected_omega(self):
        """
        Compute the expected value of omega given the expected value of psi
        """
        # We cannot assume E[\psi \psi^T] is diagonal!
        psis = self.mf_sample_psis()
        return self.b / 2.0 * (np.tanh(psis/2.0) / (psis)).mean(axis=1)

    @abc.abstractmethod
    def expected_log_likelihood(self,x):
        """
        Compute the expected log likelihood with expected parameters x
        """
        raise NotImplementedError()

    def get_vlb(self):
        # 1. E[ \ln p(s | \psi) ]
        # Compute this with Monte Carlo integration over \psi
        psis = self.mf_sample_psis()
        ps = logistic(psis)
        E_lnp = np.log(ps).mean(axis=1)
        E_ln_notp = np.log(1-ps).mean(axis=1)

        vlb = self.expected_log_likelihood((E_lnp, E_ln_notp)).sum()
        return vlb


class _NoisyPolyaGammaAugmentedCountsBase(_PolyaGammaAugmentedCountsBase):
    """
    We may want to allow for a noisy activation, psi. That is,
        psi_{t,n} = Normal(\mu_{\psi_{t,n}, \eta^2).

    To implement this, we just override psi here.
    """
    def __init__(self, X, counts, neuron):
        # Call super class constructor
        super(_NoisyPolyaGammaAugmentedCountsBase, self).__init__(X, counts, neuron)

        # Initialize noisy activation
        sigma = self.neuron.eta
        self._psi = self.neuron.mean_activation(X) + \
                    np.sqrt(sigma) * np.random.randn(self.T)

        # Initialize mean field activation
        self._mf_mu_psi = np.copy(self.psi)
        self._mf_sigma_psi = np.ones_like(self._mf_mu_psi)

    @property
    def psi(self):
        """
        Override the base class since we keep our own psi
        """
        return self._psi

    # @property
    # def mean_psi(self):
    #     return self.neuron.mean_activation(self.X)
    #
    # @property
    # def eta(self):
    #     return self.neuron.eta

    @property
    def mu_psi_prior(self):
        return self.neuron.mean_activation(self.X)

    @property
    def sigmasq_psi_prior(self):
        return self.neuron.eta

    @property
    def mu_psi_likelihood(self):
        return self.kappa / self.omega

    @property
    def sigmasq_psi_likelihood(self):
        return 1.0/self.omega

    def log_likelihood(self, x):
        return 0

    def resample(self,data=[]):
        super(_NoisyPolyaGammaAugmentedCountsBase, self).resample(data)

        # also resample psi
        self.resample_psi()

    def resample_psi(self):

        sigmasq_psi_post = 1.0 / (1.0/self.sigmasq_psi_prior + 1.0/self.sigmasq_psi_likelihood)
        mu_psi_post      = sigmasq_psi_post * \
                           (self.mu_psi_likelihood / self.sigmasq_psi_likelihood
                            + self.mu_psi_prior / self.sigmasq_psi_prior)
        self._psi = mu_psi_post + \
                    np.sqrt(sigmasq_psi_post) * np.random.normal(size=(self.T,))

    ### Mean Field
    def mf_expected_psi(self):
        """
        The second parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi).
        By default, this is the model's mean activation under the mean field
        approximation.
        """
        return self._mf_mu_psi

    def mf_marginal_variance_psi(self):
        return self._mf_sigma_psi

    def mf_covariance_psi(self):
        return np.diag(self._mf_sigma_psi)

    def mf_expected_psisq(self):
        return self._mf_sigma_psi + self._mf_mu_psi**2

    def mf_sample_psis(self, N_psis=100):
        """
        Sample psis from variational distribution
        """
        # We know the variational distributino over psi has diagonal covariance
        psis = self._mf_mu_psi[:,None] \
                + np.sqrt(self._mf_sigma_psi[:,None]) * np.random.randn(self.T, N_psis)

        return psis

    # def expected_omega(self):
    #     """
    #     Compute the expected value of omega given the expected value of psi
    #     """
    #     psis = self.mf_sample_psis()
    #     return self.b / 2.0 * (np.tanh(psis/2.0) / (psis)).mean(axis=1)

    def meanfieldupdate(self,data,weights):
        super(_NoisyPolyaGammaAugmentedCountsBase, self).meanfieldupdate(data, weights)

        self.meanfield_update_psi()

    def meanfield_update_psi(self):
        """
        Update psi and omega. We never explicitly instantiate q(omega),
        but we can compute expectations with respect to it using Monte Carlo.
        """
        # Update psi given q(omega).
        E_omega = self.expected_omega()
        mf_mean_activation = self.neuron.mf_mean_activation(self.X)
        E_eta_inv = self.neuron.noise_model.expected_eta_inv()

        self._mf_sigma_psi = 1.0/(E_omega + E_eta_inv)
        self._mf_mu_psi = self._mf_sigma_psi * (self.kappa + E_eta_inv * mf_mean_activation)

    def get_vlb(self):
        """
        Compute the variational lower bound for terms that depend on psi.
        Ignore terms that depend on omega since they are not required for
        the generative model or to compute p(S, psi, ...) or q(S, psi).
        """
        vlb = 0

        vlb += super(_NoisyPolyaGammaAugmentedCountsBase, self).get_vlb()

        # Compute the expected log prob of psi under the variational approximation for
        # the mean activation:
        # E[\ln p(psi | mu, sigma) ]
        E_psi         = self.mf_expected_psi()
        E_psisq       = self.mf_expected_psisq()
        E_mu          = self.neuron.mf_mean_activation(self.X)
        E_musq        = self.neuron.mf_expected_activation_sq(self.X)
        E_eta_inv     = self.neuron.noise_model.expected_eta_inv()
        E_ln_eta      = self.neuron.noise_model.expected_log_eta()
        vlb += ScalarGaussian().negentropy(E_x=E_psi, E_xsq=E_psisq,
                                           E_mu=E_mu, E_musq=E_musq,
                                           E_sigmasq_inv=E_eta_inv,
                                           E_ln_sigmasq=E_ln_eta).sum()

        # 3. - E[ \ln q(psi) ]
        # This is the entropy of the Gaussian distribution over psi
        vlb -= ScalarGaussian(self._mf_mu_psi, self._mf_sigma_psi).negentropy().sum()

        return vlb


class AugmentedNegativeBinomialCounts(_PolyaGammaAugmentedCountsBase):
    def __init__(self, X, counts, neuron, xi):
        super(AugmentedNegativeBinomialCounts, self).__init__(X, counts, neuron)

        assert xi > 0, "Xi must greater than 0 for negative binomial NB(xi, p)"
        self.xi = xi

    @property
    def a(self):
        return self.counts

    @property
    def b(self):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        return self.counts + self.xi

    def rvs(self, size=[]):
        p = np.exp(self.psi) / (1.0 + np.exp(self.psi))
        return np.random.rand(*p.shape) < p

    def expected_log_likelihood(self,x):
        """
        Compute the expected log likelihood with expected parameters x
        """
        E_ln_p, E_ln_notp = x
        return self.counts * E_ln_p + (1-self.counts) * E_ln_notp


class NoisyAugmentedNegativeBinomialCounts(AugmentedNegativeBinomialCounts,
                                           _NoisyPolyaGammaAugmentedCountsBase):
    pass

# We can also do logistic regression as a special case!
class AugmentedBernoulliCounts(_PolyaGammaAugmentedCountsBase):
    @property
    def a(self):
        return self.counts

    @property
    def b(self):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        return np.ones_like(self.T)

    def rvs(self, size=[]):
        p = np.exp(self.psi) / (1.0 + np.exp(self.psi))
        return np.random.rand(*p.shape) < p

    def expected_log_likelihood(self,x):
        """
        Compute the expected log likelihood with expected parameters x
        """
        E_ln_p, E_ln_notp = x
        return self.counts * E_ln_p + (1-self.counts) * E_ln_notp

    # def mf_expected_psi(self):
    #     return self.mf_mu_psi
    #
    # def mf_expected_psisq(self):
    #     return self.mf_sigma_psi + self.mf_mu_psi**2
    #
    # def meanfield_update_psi(self):
    #     """
    #     Update psi and omega. We never explicitly instantiate q(omega),
    #     but we can compute expectations with respect to it using Monte Carlo.
    #     """
    #     # Update psi given q(omega). Use Monte Carlo to approximate the expectation of
    #     # omega over 100 samples of psi
    #     ccounts = self.counts - 0.5
    #     psis = self.mf_mu_psi[:,None] + \
    #              np.sqrt(self.mf_sigma_psi)[:,None] * np.random.randn(self.T, 100)
    #     E_omega = (np.tanh(psis/2.0) / (2*psis)).mean(axis=1)
    #
    #     mf_mean_activation = self.neuron.mf_mean_activation(self.X)
    #
    #     E_eta_inv = self.neuron.noise_model.expected_eta_inv()
    #     self.mf_sigma_psi = 1.0/(E_omega + E_eta_inv)
    #     self.mf_mu_psi = self.mf_sigma_psi * (ccounts +
    #                                           E_eta_inv * mf_mean_activation)

class NoisyAugmentedBernoulliCounts(AugmentedBernoulliCounts,
                                    _NoisyPolyaGammaAugmentedCountsBase):
    pass

# Finally, support the standard Poisson observations, but to be
# consistent with the NB and Bernoulli models we add a bit of
# Gaussian noise to the activation, psi.
class _LinearNonlinearPoissonCountsBase(GibbsSampling):
    """
    Counts s ~ Poisson(log(1+f(psi))) where f is a rectifying nonlinearity.
    """
    def __init__(self, X, counts, neuron, nsteps=3, step_sz=0.1):
        assert counts.ndim == 1
        self.counts = counts.astype(np.int32)
        self.T = counts.shape[0]

        # assert X.ndim == 2 and X.shape[0] == self.T
        self.X = X

        # Keep this pointer to the model
        self.model = neuron

        # Initialize the activation
        self.psi = self.model.mean_activation(X)

        # Set the number of HMC steps
        self.nsteps = nsteps
        self.step_sz = step_sz

    def f(self, x):
        """
        Return the rate for a given activation x
        """
        raise NotImplementedError()

    def grad_f(self, x):
        """
        Return the nonlinear function of psi
        """
        raise NotImplementedError()

    def rvs(self, size=[]):
        return None


    def log_likelihood(self, x):
        """
        Return the the log likelihood of counts given activation x
        """
        rate = self.f(x)
        return self.counts * np.log(rate) - rate

    def grad_log_likelihood(self, x):
        """
        Return the gradient of the log likelihood of counts given activation x
        """
        rate = self.f(x)
        grad_rate = self.grad_f(x)

        return self.counts / rate * grad_rate - grad_rate

    def log_posterior_psi(self, x):
        mu_prior = self.model.mean_activation(self.X)
        sigma_prior = self.model.eta

        return -0.5/sigma_prior * (x-mu_prior)**2 + self.log_likelihood(x)

    def grad_log_posterior_psi(self, x):
        mu_prior = self.model.mean_activation(self.X)
        sigma_prior = self.model.eta

        return -1.0/sigma_prior * (x-mu_prior) + self.grad_log_likelihood(x)

    def resample(self, data=None, stats=None,
                 do_resample_psi=True,
                 do_resample_aux=True):
        """
        Resample the activation psi given the counts and the model prior
        using Hamiltonian Monte Carlo
        """
        if not do_resample_psi:
            return

        psi_orig = self.psi
        nsteps = self.nsteps
        step_sz = self.step_sz

        # Start at current state
        psi = np.copy(psi_orig)
        # Momentum is simplest for a normal rv
        p = np.random.randn(*np.shape(psi))
        p_curr = np.copy(p)

        # Set a prefactor of -1 since we're working with log probs
        pre = -1.0

        # Evaluate potential and kinetic energies at start of trajectory
        U_curr = pre * self.log_posterior_psi(psi_orig)
        K_curr = np.sum(p_curr**2)/2.0

        # Make a half step in the momentum variable
        p -= step_sz * pre * self.grad_log_posterior_psi(psi)/2.0

        # Alternate L full steps for position and momentum
        for i in np.arange(self.nsteps):
            psi += step_sz*p

            # Full step for momentum except for last iteration
            if i < nsteps-1:
                p -= step_sz * pre * self.grad_log_posterior_psi(psi)
            else:
                p -= step_sz * pre * self.grad_log_posterior_psi(psi)/2.0

        # Negate the momentum at the end of the trajectory to make proposal symmetric?
        p = -p

        # Evaluate potential and kinetic energies at end of trajectory
        U_prop = pre * self.log_posterior_psi(psi)
        K_prop = p**2/2.0

        # Accept or reject new state with probability proportional to change in energy.
        # Ideally this will be nearly 0, but forward Euler integration introduced errors.
        # Exponentiate a value near zero and get nearly 100% chance of acceptance.
        not_accept = np.log(np.random.rand(*psi.shape)) > U_curr-U_prop + K_curr-K_prop
        psi[not_accept] = psi_orig[not_accept]

        self.psi = psi


class ReLuPoissonCounts(_LinearNonlinearPoissonCountsBase):
    """
    Rectified linear Poisson counts.
    """
    def f(self, x):
        return np.log(1.0+np.exp(x))

    def grad_f(self, x):
        return np.exp(x)/(1.0+np.exp(x))


class ExpPoissonCounts(_LinearNonlinearPoissonCountsBase):
    """
    Rectified linear Poisson counts.
    """
    def f(self, x):
        return np.exp(x)

    def grad_f(self, x):
        return np.exp(x)