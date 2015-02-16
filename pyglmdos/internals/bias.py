"""
Bias models (of which there are only one)
"""
import numpy as np

from pyglmdos.abstractions import Component

class GaussianBias(Component):

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

    def resample(self, augmented_data):
        """
        Resample the bias given the weights and psi
        :return:
        """
        self._resample_b()

    def _resample_b(self, augmented_data):
        # TODO: Parallelize this
        for n in xrange(self.N):
            # Compute the posterior parameters
            lkhd_prec           = self.activation.precision(augmented_data, bias=n)
            lkhd_mean_dot_prec  = self.activation.mean_dot_precision(augmented_data, bias=n)

            prior_prec          = self.lambda_0
            prior_mean_dot_prec = self.lambda_0 * self.mu_0

            post_prec           = prior_prec + lkhd_prec
            post_mu             = 1.0 / post_prec * (prior_mean_dot_prec + lkhd_mean_dot_prec)

            self.b[n] = post_mu + np.sqrt(1.0/post_prec) * np.random.randn()