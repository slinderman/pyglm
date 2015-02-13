import numpy as np

from pyglm.deps.pybasicbayes.abstractions import MeanField, GibbsSampling
from pyglm.deps.pybasicbayes.util.stats import getdatasize
from pyglm.internals.distributions import ScalarGaussian

class GaussianBias(GibbsSampling, MeanField):
    def __init__(self,
                 neuron_model,
                 mu, sigmasq):
        self.neuron_model = neuron_model
        self.mu_0 = mu
        self.sigmasq = sigmasq
        self.lambda_0 = 1.0/self.sigmasq

        # Initialize mean field parameters
        self.mf_mu_bias = self.mu_0
        self.mf_sigma_bias = self.sigmasq

        # Initialize the bias from the prior
        self.resample()

    @property
    def sigma_inv(self):
        return 1.0 / self.neuron_model.noise_model.sigma

    def log_likelihood(self,x):
        return ScalarGaussian(self.mu_0, self.sigmasq).log_probability(x)

    def log_probability(self):
        return self.log_likelihood(self.bias)

    def rvs(self,size=[]):
        raise NotImplementedError()

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = data.mean(0)
            else:
                xbar = sum(d.sum(0) for d in data) / n
        else:
            xbar = None

        return n, xbar

    def _posterior_hypparams(self,n,xbar):
        # It seems we should be working with lmbda and sigma inv (unless lmbda is a covariance, not a precision)
        sigma_inv, mu_0, lambda_0 = self.sigma_inv, self.mu_0, self.lambda_0
        if n > 0:
            lambda_n = n*sigma_inv + lambda_0
            # mu_n = np.linalg.solve(sigma_inv_n, sigma_inv_0.dot(mu_0) + n*sigma_inv.dot(xbar))
            mu_n = (lambda_0 * mu_0 + n * sigma_inv * xbar) / lambda_n
            return mu_n, lambda_n
        else:
            return mu_0, lambda_0

    def resample(self):
        """
        Resample the bias given the weights and psi
        :return:
        """
        # Compute the posterior parameters
        lkhd_prec           = self.neuron_model.activation_lkhd_precision(0)
        lkhd_mean_dot_prec  = self.neuron_model.activation_lkhd_mean_dot_precision(0)

        prior_prec          = self.lambda_0
        prior_mean_dot_prec = self.lambda_0 * self.mu_0

        post_prec           = prior_prec + lkhd_prec
        post_mu             = 1.0 / post_prec * (prior_mean_dot_prec + lkhd_mean_dot_prec)

        self.bias = post_mu + np.sqrt(1.0/post_prec) * np.random.randn()

    def expected_log_likelihood(self,x):
        # TODO
        raise NotImplementedError()

    def mf_expected_bias(self):
        return self.mf_mu_bias

    def mf_expected_bias_sq(self):
        # Var(bias) = E[bias^2] - E[bias]^2
        return self.mf_sigma_bias + self.mf_mu_bias**2

    def meanfieldupdate(self):
        """
        Update the variational parameters for the bias
        """
        if len(self.neuron_model.data_list) > 0:
            residuals = []
            for d in self.neuron_model.data_list:
                mu = np.zeros_like(d.psi)
                for X,syn in zip(d.X, self.neuron_model.synapse_models):
                    mu += syn.mf_predict(X)

                # Use mean field activation to compute residuals
                residual = (d.mf_expected_psi() - mu)[:,None]
                residuals.append(residual)
            residuals = np.vstack(residuals)

            # TODO: USE MF ETA to compute residual
            T = residuals.shape[0]
            E_eta_inv = self.neuron_model.noise_model.expected_eta_inv()
            self.mf_sigma_bias = 1.0/(T * E_eta_inv + 1.0/self.sigmasq)
            self.mf_mu_bias = self.mf_sigma_bias * (residuals.sum() * E_eta_inv +
                                                    self.mu_0/self.sigmasq)

            self.mf_sigma_bias = self.mf_sigma_bias
            self.mf_mu_bias = self.mf_mu_bias

        else:
            self.mf_sigma_bias = self.sigmasq
            self.mf_mu_bias = self.mu_0

    def get_vlb(self):
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

        vlb += ScalarGaussian(self.mu_0, self.sigmasq).negentropy(E_x=E_b,
                                                                  E_xsq=E_bsq).sum()

        # Second term
        # E[LN q(b | \tilde{\mu}, \tilde{\sigma^2})]
        vlb -= ScalarGaussian(self.mf_mu_bias, self.mf_sigma_bias).negentropy().sum()

        return vlb