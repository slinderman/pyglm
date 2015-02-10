import numpy as np
from scipy.special import gammaln

from hips.distributions.polya_gamma import polya_gamma

from pyglm.deps.pybasicbayes.abstractions import GibbsSampling, MeanField
from pyglm.internals.distributions import ScalarGaussian
from pypolyagamma import pgdrawv, PyRNG

from pyglm.utils.utils import logistic

from hips.inference.log_sum_exp import log_sum_exp_sample

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
        sigma = self.model.eta
        self.psi = self.model.mean_activation(X) + \
                   np.sqrt(sigma) * np.random.randn(self.T)
        #
        # self.omega = polya_gamma(np.ones(self.T),
        #                          self.psi.reshape(self.T),
        #                          200).reshape((self.T,))
        self.omega = np.ones(self.T)
        rng = PyRNG()
        pgdrawv(np.ones(self.T, dtype=np.int32), self.psi, self.omega, rng)

        # Initialize mean field activation
        self.mf_mu_psi = np.copy(self.psi)
        self.mf_sigma_psi = np.ones_like(self.mf_mu_psi)

    def log_likelihood(self, x):
        return 0

    def rvs(self, size=[]):
        return None

class AugmentedNegativeBinomialCounts(_PolyaGammaAugmentedCountsBase):

    def resample(self, data=None, stats=None,
                 do_resample_psi=True,
                 do_resample_aux=True,
                 do_resample_psi_from_prior=False):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        xi = np.int32(self.model.xi)
        mu = self.model.mean_activation(self.X)
        sigma = self.model.eta

        if do_resample_aux:
            # Resample the auxiliary variables, omega, in Python
            # self.omega = polya_gamma(self.counts.reshape(self.T)+xi,
            #                          self.psi.reshape(self.T),
            #                          200).reshape((self.T,))

            # Create a PyPolyaGamma object and resample with the C code
            # seed = np.random.randint(2**16)
            # ppg = PyPolyaGamma(seed, self.model.trunc)
            # ppg.draw_vec(self.counts+xi, self.psi, self.omega)

            # Resample with Jesse Windle's ported code
            rng = PyRNG()
            pgdrawv(self.counts+xi, self.psi, self.omega, rng)

        # Resample the rates, psi given omega and the regression parameters
        if do_resample_psi:
            sig_post = 1.0 / (1.0/sigma + self.omega)
            mu_post = sig_post * ((self.counts-xi)/2.0 + mu / sigma)
            self.psi = mu_post + np.sqrt(sig_post) * np.random.normal(size=(self.T,))

        # For Geweke testing, just resample psi from the forward model
        elif do_resample_psi_from_prior:
            mu_prior = self.model.mean_activation(self.X)
            sigma_prior = self.model.eta
            self.psi = mu_prior + np.sqrt(sigma_prior) * np.random.normal(size=(self.T,))

    def geweke_resample_counts(self, trunc=100):
        """
        Resample the counts given omega and psi.
        Given omega, the distribution over y is no longer negative binomial.
        Instead, it takes a pretty ugly form. We have,
        log p(y | xi, psi, omega) = c + log Gamma(y+xi) - log y! - y + psi * (y-xi)/2
        """
        xi = self.model.xi
        ys = np.arange(trunc)[:,None]
        lp = gammaln(ys+xi) - gammaln(ys+1) - ys + (ys-xi) / 2.0 * self.psi[None,:]
        self.counts = log_sum_exp_sample(lp, axis=0)

    def meanfield_update_psi(self):
        """
        Update psi and omega. We never explicitly instantiate q(omega),
        but we can compute expectations with respect to it using Monte Carlo.
        """
        # Update psi given q(omega). Use Monte Carlo to approximate the expectation of
        # omega over 100 samples of psi
        ccounts = (self.counts + self.xi)/2.0
        psis = self.mf_mu_psi[:,None] + \
                 np.sqrt(self.mf_sigma_psi)[:,None] * np.random.randn(self.T, 100)
        E_omega = ccounts * (np.tanh(psis/2.0) / (psis)).mean(axis=1)

        # TODO: Use MF ETA
        mf_eta = self.model.eta
        self.mf_sigma_psi = 1.0/(E_omega + 1.0/mf_eta)
        self.mf_mu_psi = self.mf_sigma_psi * (ccounts +
                                              1.0/mf_eta * self.model.mf_mean_activation(self.X))


# We can also do logistic regression as a special case!
class AugmentedBernoulliCounts(_PolyaGammaAugmentedCountsBase):
    def resample(self, data=None, stats=None,
                 do_resample_psi=True,
                 do_resample_psi_from_prior=False,
                 do_resample_aux=True):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        if do_resample_aux:
            # Resample the auxiliary variables, omega, in Python
            # self.omega = polya_gamma(np.ones(self.T),
            #                          self.psi.reshape(self.T),
            #                          200).reshape((self.T,))

            # Resample with the C code
            # Create a PyPolyaGamma object
            # seed = np.random.randint(2**16)
            # ppg = PyPolyaGamma(seed, self.model.trunc)
            # ppg.draw_vec(np.ones(self.T, dtype=np.int32), self.psi, self.omega)

            # Resample with Jesse Windle's code
            rng = PyRNG()
            pgdrawv(np.ones(self.T, dtype=np.int32), self.psi, self.omega, rng)

        # Resample the rates, psi given omega and the regression parameters
        if do_resample_psi:
            mu_prior = self.model.mean_activation(self.X)
            sigma_prior = self.model.eta

            sig_post = 1.0 / (1.0/sigma_prior + self.omega)
            mu_post = sig_post * (self.counts-0.5 + mu_prior / sigma_prior)
            self.psi = mu_post + np.sqrt(sig_post) * np.random.normal(size=(self.T,))

        # For Geweke testing, just resample psi from the forward model
        elif do_resample_psi_from_prior:
            mu_prior = self.model.mean_activation(self.X)
            sigma_prior = self.model.eta
            self.psi = mu_prior + np.sqrt(sigma_prior) * np.random.normal(size=(self.T,))

    # def cond_omega(self):
    #     """
    #     Compute the conditional distribution of omega given the counts and psi
    #     :return:
    #     """
    #     # TODO: Finish this for unit testing
    #     return PolyaGamma(np.ones(self.T, dtype=np.int32), self.psi)

    # def cond_psi(self):
    #     """
    #     Compute the conditional distribution of psi given the counts and omega
    #     :return:
    #     """
    #     # TODO: Finish this for unit testing
    #     mu_prior = self.model.mean_activation(self.X)
    #     sigma_prior = self.model.eta
    #
    #     sig_post = 1.0 / (1.0/sigma_prior + self.omega)
    #     mu_post = sig_post * (self.counts-0.5 + mu_prior / sigma_prior)
    #     return DiagonalGaussian(mu_post, sig_post)

    def geweke_resample_counts(self):
        """
        Resample the counts given omega and psi.
        Given omega, the distribution over y is no longer negative binomial.
        Instead, it takes a pretty ugly form. We have,
        log p(y | xi, psi, omega) = c + log Gamma(y+xi) - log y! - y + psi * (y-xi)/2
        """
        ys = np.arange(2)[None,:]
        psi = self.psi[:,None]
        # omega = self.omega[:,None]
        # lp = -np.log(2.0) + (ys-0.5) * psi - omega * psi**2 / 2.0
        lp = (ys-0.5) * psi
        for t in xrange(self.T):
            self.counts[t] = log_sum_exp_sample(lp[t,:])


    def rvs(self, size=[]):
        p = np.exp(self.psi) / (1.0 + np.exp(self.psi))
        return np.random.rand(*p.shape) < p

    def mf_expected_psi(self):
        return self.mf_mu_psi

    def mf_expected_psisq(self):
        return self.mf_sigma_psi + self.mf_mu_psi**2

    def meanfield_update_psi(self):
        """
        Update psi and omega. We never explicitly instantiate q(omega),
        but we can compute expectations with respect to it using Monte Carlo.
        """
        # Update psi given q(omega). Use Monte Carlo to approximate the expectation of
        # omega over 100 samples of psi
        ccounts = self.counts - 0.5
        psis = self.mf_mu_psi[:,None] + \
                 np.sqrt(self.mf_sigma_psi)[:,None] * np.random.randn(self.T, 100)
        E_omega = (np.tanh(psis/2.0) / (2*psis)).mean(axis=1)

        mf_mean_activation = self.model.mf_mean_activation(self.X)

        E_eta_inv = self.model.noise_model.expected_eta_inv()
        self.mf_sigma_psi = 1.0/(E_omega + E_eta_inv)
        self.mf_mu_psi = self.mf_sigma_psi * (ccounts +
                                              E_eta_inv * mf_mean_activation)

    def get_vlb(self):
        """
        Compute the variational lower bound for terms that depend on psi.
        Ignore terms that depend on omega since they are not required for
        the generative model or to compute p(S, psi, ...) or q(S, psi).
        """
        vlb = 0
        # 1. E[ \ln p(s | \psi) ]
        # Compute this with Monte Carlo integration over \psi
        N_mc = 100
        psis = self.mf_mu_psi[:,None] + \
                 np.sqrt(self.mf_sigma_psi)[:,None] * np.random.randn(self.T, N_mc)
        ps = logistic(psis)
        E_lnp = np.log(ps).mean(axis=1)
        E_ln_notp = np.log(1-ps).mean(axis=1)

        vlb += (self.counts * E_lnp + (1-self.counts) * E_ln_notp).sum()

        # 2. E[\ln p(psi | mu, sigma) ]
        # Compute the expected log prob of psi under the variational approximation for
        # the mean activation
        # TODO: Compute expectations wrt variational distribution
        E_psi         = self.mf_expected_psi()
        E_psisq       = self.mf_expected_psisq()
        E_mu          = self.model.mf_mean_activation(self.X)
        E_musq        = self.model.mf_expected_activation_sq(self.X)
        E_eta_inv     = self.model.noise_model.expected_eta_inv()
        E_ln_eta      = self.model.noise_model.expected_log_eta()
        vlb += ScalarGaussian().negentropy(E_x=E_psi, E_xsq=E_psisq,
                                           E_mu=E_mu, E_musq=E_musq,
                                           E_sigmasq_inv=E_eta_inv,
                                           E_ln_sigmasq=E_ln_eta).sum()

        # 3. - E[ \ln q(psi) ]
        # This is the entropy of the Gaussian distribution over psi
        vlb -= ScalarGaussian(self.mf_mu_psi, self.mf_sigma_psi).negentropy().sum()

        return vlb


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