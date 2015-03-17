import abc
import numpy as np
from scipy.special import gammaln

# from hips.distributions.polya_gamma import polya_gamma
from pypolyagamma import pgdrawv, PyRNG

from hips.inference.slicesample import slicesample

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
        self.N = self.population.N

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
        T = S.shape[0]
        assert S.shape[1] == self.N

        # Initialize auxiliary variables
        augmented_data["omega"] = np.empty((T, self.N))
        for n in xrange(self.N):
            tmp = np.empty(T)
            pgdrawv(np.ones(T, dtype=np.int32),
                    np.zeros(T),
                    tmp, self.rng)
            augmented_data["omega"][:,n] = tmp

        # Precompute kappa (assuming that it is constant given data)
        # That is, we can only do this if xi is not resampled
        augmented_data["kappa"] = self.a(augmented_data) - self.b(augmented_data)/2.0

        # Initialize the mean field local variational parameters
        augmented_data["omega"] = np.empty((T, self.N))

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
        return self.a(augmented_data) - self.b(augmented_data)/2.0
        # return augmented_data["kappa"]

    def omega(self, augmented_data):
        return augmented_data["omega"]

    @abc.abstractmethod
    def rvs(self, Psi):
        raise NotImplementedError()

    @abc.abstractmethod
    def expected_S(self, Psi):
        raise NotImplementedError()

    def resample(self, augmented_data_list):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        for augmented_data in augmented_data_list:
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
        Psis = self.activation.mf_sample_marginal_activation(augmented_data, N_samples=20)
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
        # Psis = self.activation.mf_sample_activation(augmented_data, N_samples=1)
        Psis = self.activation.mf_sample_marginal_activation(augmented_data, N_samples=10)
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
    def __init__(self, population, xi=None, alpha_xi=None, beta_xi=None):
        super(NegativeBinomialObservations, self).__init__(population)

        if alpha_xi is not None and beta_xi is not None:
            self.do_resample_xi = True
            self.alpha_xi = alpha_xi
            self.beta_xi = beta_xi

            if xi is None:
                # We use xi = 1 + Gamma(alpha, beta)
                self.xi = 1 + np.random.gamma(alpha_xi, 1./beta_xi, size=(1,self.N))

        if xi is not None:
            if np.isscalar(xi):
                assert xi > 0, "Xi must greater than 0 for negative binomial NB(xi, p)"
                self.xi = xi * np.ones((1,self.N))
            else:
                assert xi.shape == (1,self.N) and np.amin(xi) >= 0
                self.xi = xi

            if alpha_xi is None and beta_xi is None:
                self.do_resample_xi = False

        if alpha_xi is None and beta_xi is None and xi is None:
            raise Exception("Either alpha_xi or beta_xi must be specified")

    # @property
    # def N(self):
    #     return self.population.N

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
        # if "_b" in augmented_data:
        #     return augmented_data["_b"]
        # else:
        return (augmented_data["S"] + self.xi).astype(np.int32)

    def rvs(self, Psi):
        p = logistic(Psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        return np.random.negative_binomial(self.xi, 1-p)

    # Override the Gibbs sampler to also sample xi
    def resample(self, augmented_data_list):
        if self.do_resample_xi:
            self._resample_xi_discrete(augmented_data_list)
            # self._resample_xi_slicesample(augmented_data_list)

        super(NegativeBinomialObservations, self).resample(augmented_data_list)

        # Save b
        # for data in augmented_data_list:
        #     data["_b"] = (data["S"] + self.xi).astype(np.int32)

    def _resample_xi_slicesample(self, augmented_data_list):
        # Compute the activations
        Ss   = np.vstack([d["S"] for d in augmented_data_list])
        psis = np.vstack([self.activation.compute_psi(d) for d in augmented_data_list])

        # Resample xi using slice sampling
        # p(\xi | \psi, s) \propto p(\xi) * p(s | \xi, \psi)
        for n in xrange(self.N):
            Sn   = Ss[:,n]
            psin = psis[:,n]
            pn   = logistic(psin)
            pn   = np.clip(pn, 1e-32, 1-1e-32)

            def _log_prob_xin(xin):
                lp = 0

                # Compute the prior of \xi_n ~ 1 + Gamma(alpha, beta)
                assert xin > 1
                lp += (self.alpha_xi-1) * np.log(xin-1) - self.beta_xi * (xin-1)

                # Compute the likelihood of \xi_n, NB(S_{t,n} | xi_n, \psi_{t,n})
                lp += (gammaln(Sn+xin) - gammaln(xin)).sum()
                lp += (xin * np.log(1-pn)).sum()

                return lp

            # Slice sample \xi_n
            self.xi[0,n], _ = slicesample(self.xi[0,n], _log_prob_xin, lb=1+1e-5, ub=100)

        print "Xi:"
        print self.xi

    def _resample_xi_discrete(self, augmented_data_list, xi_max=100):
        # Compute the activations
        Ss   = np.vstack([d["S"] for d in augmented_data_list])
        psis = np.vstack([self.activation.compute_psi(d) for d in augmented_data_list])

        # Resample xi using slice sampling
        # p(\xi | \psi, s) \propto p(\xi) * p(s | \xi, \psi)
        for n in xrange(self.N):
            Sn   = Ss[:,n]
            psin = psis[:,n]
            pn   = logistic(psin)
            pn   = np.clip(pn, 1e-32, 1-1e-32)

            from hips.inference.log_sum_exp import log_sum_exp_sample
            xis = np.arange(1, xi_max)
            lp_xi = (gammaln(Sn[:,None]+xis[None,:]) - gammaln(xis[None,:])).sum(0)
            lp_xi += (xis[None,:] * np.log(1-pn)[:,None]).sum(0)
            self.xi[0,n] = xis[log_sum_exp_sample(lp_xi)]

        print self.xi

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