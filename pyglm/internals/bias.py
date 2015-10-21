"""
Bias models (of which there are only one)
"""
import numpy as np

from pyglm.abstractions import Component
from pyglm.internals.distributions import ScalarGaussian

class _GaussianBiasBase(Component):

    def __init__(self, population, mu_0=0.0, sigma_0=1.0):
        self.population = population

        # Save the prior parameters
        self.mu_0, self.sigma_0 = mu_0, sigma_0
        self.lambda_0 = 1.0/self.sigma_0

        # Initialize the parameters
        self.b = np.zeros((self.N,))

    @property
    def N(self):
        return self.population.N

    @property
    def activation(self):
        return self.population.activation_model

    def initialize_with_standard_model(self, standard_model):
        # Initialize the biases
        b_std = standard_model.bias.copy()
        assert b_std.shape == (self.N,)
        self.b = b_std

        # Also initialize the bias distribution
        self.mu_0 = b_std.mean()
        self.lambda_0 = 1./b_std.var()

    def log_prior(self):
        return ScalarGaussian(self.mu_0, self.sigma_0).log_probability(self.b).sum()

    def sample_predictive_distribution(self):
        """
        Sample a new bias from the prior
        :return:
        """
        return self.mu_0 + np.sqrt(self.sigma_0) * np.random.randn()


class _GibbsGaussianBias(_GaussianBiasBase):
    def __init__(self, population, mu_0=0.0, sigma_0=1.0):
        super(_GibbsGaussianBias, self).__init__(population, mu_0=mu_0, sigma_0=sigma_0)

        # Initialize with a draw from the prior
        self.resample()

    def resample(self, augmented_data=[]):
        """
        Resample the bias given the weights and psi
        :return:
        """
        self._resample_b(augmented_data)

    def _resample_b(self, augmented_data_list):


        # TODO: Parallelize this
        for n in xrange(self.N):
            # Compute the posterior parameters
            lkhd_prec           = 0
            lkhd_mean_dot_prec  = 0

            for augmented_data in augmented_data_list:
                lkhd_prec           += self.activation.precision(augmented_data, bias=n)
                lkhd_mean_dot_prec  += self.activation.mean_dot_precision(augmented_data, bias=n)

            prior_prec          = self.lambda_0
            prior_mean_dot_prec = self.lambda_0 * self.mu_0

            post_prec           = prior_prec + lkhd_prec
            post_mu             = 1.0 / post_prec * (prior_mean_dot_prec + lkhd_mean_dot_prec)

            self.b[n] = post_mu + np.sqrt(1.0/post_prec) * np.random.randn()

    def collapsed_resample(self, augmented_data=[]):
        # Collapsed resampling is taken care of by the weights
        pass


class _MeanFieldGaussianBias(_GaussianBiasBase):
    def __init__(self, population, mu_0=0.0, sigma_0=1.0):
        super(_MeanFieldGaussianBias, self).__init__(population, mu_0=mu_0, sigma_0=sigma_0)

        # Initialize mean field parameters
        self.mf_mu_b    = mu_0 * np.ones(self.N)
        self.mf_sigma_b = sigma_0 * np.ones(self.N)

    def initialize_with_standard_model(self, standard_model):
        super(_MeanFieldGaussianBias, self).\
            initialize_with_standard_model(standard_model)

        b_std = standard_model.bias.copy()
        assert b_std.shape == (self.N,)

        self.mf_mu_b = b_std
        self.mf_sigma_b = 1e-6 * np.ones(self.N)

    def mf_expected_bias(self):
        return self.mf_mu_b

    def mf_expected_bias_sq(self):
        # Var(bias) = E[bias^2] - E[bias]^2
        return self.mf_sigma_b + self.mf_mu_b**2

    def meanfieldupdate(self, augmented_data):
        """
        Mean field update of the bias given the weights and psi
        :return:
        """
        for n in xrange(self.N):
            stats = self.get_expected_suff_stats(augmented_data, n)
            self._meanfieldupdate_b(n, stats)

    def get_expected_suff_stats(self, augmented_data, n, minibatchfrac=1.0):
        # Compute the expected posterior parameters
        lkhd_prec           = self.activation.mf_precision(augmented_data, bias=n)
        lkhd_mean_dot_prec  = self.activation.mf_mean_dot_precision(augmented_data, bias=n)

        prior_prec          = self.lambda_0
        prior_mean_dot_prec = self.lambda_0 * self.mu_0

        post_prec           = prior_prec + lkhd_prec / minibatchfrac
        post_mu             = 1.0 / post_prec * (prior_mean_dot_prec + lkhd_mean_dot_prec / minibatchfrac)

        return post_mu, post_prec

    def _meanfieldupdate_b(self, n, stats, stepsize=1.0):
        post_mu, post_prec = stats
        self.mf_mu_b[n]    = (1-stepsize) * self.mf_mu_b[n] \
                             + stepsize * post_mu
        self.mf_sigma_b[n] = (1-stepsize) * self.mf_sigma_b[n] \
                             + stepsize * 1.0 / post_prec

    def get_vlb(self, augmented_data):
        """
        Variational lower bound for the Gaussian bias
        E[LN p(b | \mu, \sigma^2)] -
        E[LN q(b | \tilde{\mu}, \tilde{\sigma^2})]
        :return:
        """
        vlb = 0

        # First term
        # E[LN p(b | mu, sigma^2)]
        E_b   = self.mf_expected_bias()
        E_bsq = self.mf_expected_bias_sq()

        vlb += ScalarGaussian(self.mu_0, self.sigma_0).negentropy(E_x=E_b,
                                                                  E_xsq=E_bsq).sum()

        # Second term
        # E[LN q(b | \tilde{\mu}, \tilde{\sigma^2})]
        vlb -= ScalarGaussian(self.mf_mu_b, self.mf_sigma_b).negentropy().sum()

        return vlb

    def resample_from_mf(self, augmented_data):
        """
        Resample from the variational distribution
        """
        self.b = self.mf_mu_b + np.sqrt(self.mf_sigma_b) * np.random.randn(self.N)

    def mf_mode(self):
        self.b = self.mf_mu_b.copy()

    ### SVI
    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        for n in xrange(self.N):
            stats = self.get_expected_suff_stats(augmented_data, n,
                                                 minibatchfrac=minibatchfrac)
            self._meanfieldupdate_b(n, stats, stepsize=stepsize)


class GaussianBias(_GibbsGaussianBias, _MeanFieldGaussianBias):
    pass