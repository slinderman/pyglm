"""
Weight models
"""
import numpy as np

from pyglmdos.abstractions import Component

from pyglm.utils.utils import logistic, logit

class SpikeAndSlabGaussianWeights(Component):

    def __init__(self, population):
        self.population = population

        # Initialize the parameters
        self.A = np.zeros((self.N, self.N))
        self.W = np.zeros((self.N, self.N, self.B))

    @property
    def N(self):
        return self.population.N

    @property
    def B(self):
        return self.population.B

    @property
    def activation(self):
        return self.population.activation_model

    def resample(self, augmented_data):
        for n_pre in xrange(self.N):
            #  TODO: We can parallelize over n_post
            for n_post in xrange(self.N):
                stats = self._get_sufficient_statistics(augmented_data, n_pre, n_post)

                # Sample the slab variable
                if self.A[n_pre, n_post]:
                    self._resample_W(n_pre, n_post, stats)
                else:
                    self.W[n_pre, n_post,:] = 0.0

    def _get_sufficient_statistics(self, augmented_data, n_pre, n_post):
        """
        Get the sufficient statistics for this synapse.
        """
        prior_prec          = np.linalg.inv(self.Sigma_w)
        prior_mean_dot_prec = self.mu_w.dot(prior_prec)

        # Compute the posterior parameters
        lkhd_prec           = self.activation.precision(augmented_data, synapse=(n_pre,n_post))
        lkhd_mean_dot_prec  = self.activation.mean_dot_precision(augmented_data, synapse=(n_pre,n_post))

        post_prec           = prior_prec + lkhd_prec
        post_cov            = np.linalg.inv(post_prec)
        post_mu             = (prior_mean_dot_prec + lkhd_mean_dot_prec).dot(post_cov)
        post_mu             = post_mu.ravel()

        return post_mu, post_cov, post_prec

    def _resample_A(self, n_pre, n_post, stats):
        post_mu, post_cov, post_prec = stats
        rho = self.rho

        # Compute log Pr(A=0|...) and log Pr(A=1|...)
        lp_A = np.zeros(2)
        lp_A[0] = np.log(1.0-rho)
        lp_A[1] = np.log(rho)

        logdet_prior_cov = np.linalg.slogdet(self.Sigma_w)[1]
        logdet_post_cov  = np.linalg.slogdet(post_cov)[1]
        logit_rho_post   = logit(self.rho) \
                           + self.D_in / 2.0 * (logdet_post_cov - logdet_prior_cov) \
                           + 0.5 * post_mu.dot(post_prec).dot(post_mu) \
                           - 0.5 * self.mu_w.dot(np.linalg.solve(self.Sigma_w, self.mu_w))

        rho_post = logistic(logit_rho_post)

        # Sample the binary indicator of an edge
        self.A[n_pre, n_post] = np.random.rand() < rho_post

    def _resample_W(self, n_pre, n_post, stats):
        post_mu, post_cov, post_prec = stats

        self.W[n_pre, n_post, :] = np.random.multivariate_normal(post_mu, post_cov)

