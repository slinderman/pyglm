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

        self.resample(None)

    @property
    def N(self):
        return self.population.N

    @property
    def B(self):
        return self.population.B

    @property
    def W_effective(self):
        return self.A[:,:,None] * self.W

    @property
    def network(self):
        return self.population.network

    @property
    def activation(self):
        return self.population.activation_model

    def resample(self, augmented_data):
        for n_pre in xrange(self.N):
            #  TODO: We can parallelize over n_post
            for n_post in xrange(self.N):
                stats = self._get_sufficient_statistics(augmented_data, n_pre, n_post)

                # Sample the spike variable
                self._resample_A(n_pre, n_post, stats)

                # Sample the slab variable
                if self.A[n_pre, n_post]:
                    self._resample_W(n_pre, n_post, stats)
                else:
                    self.W[n_pre, n_post,:] = 0.0

    def _get_sufficient_statistics(self, augmented_data, n_pre, n_post):
        """
        Get the sufficient statistics for this synapse.
        """
        mu_w                = self.network.Mu[n_pre, n_post, :]
        Sigma_w             = self.network.Sigma[n_pre, n_post, :, :]

        prior_prec          = np.linalg.inv(Sigma_w)
        prior_mean_dot_prec = mu_w.dot(prior_prec)

        # Compute the posterior parameters
        if augmented_data is not None:
            lkhd_prec           = self.activation.precision(augmented_data, synapse=(n_pre,n_post))
            lkhd_mean_dot_prec  = self.activation.mean_dot_precision(augmented_data, synapse=(n_pre,n_post))
        else:
            lkhd_prec           = 0
            lkhd_mean_dot_prec  = 0

        post_prec           = prior_prec + lkhd_prec
        post_cov            = np.linalg.inv(post_prec)
        post_mu             = (prior_mean_dot_prec + lkhd_mean_dot_prec).dot(post_cov)
        post_mu             = post_mu.ravel()

        return post_mu, post_cov, post_prec

    def _resample_A(self, n_pre, n_post, stats):
        """
        Resample the presence or absence of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        mu_w                         = self.network.Mu[n_pre, n_post, :]
        Sigma_w                      = self.network.Sigma[n_pre, n_post, :, :]
        post_mu, post_cov, post_prec = stats
        rho                          = self.network.P[n_pre, n_post]

        # Compute the log odds ratio
        logdet_prior_cov = np.linalg.slogdet(Sigma_w)[1]
        logdet_post_cov  = np.linalg.slogdet(post_cov)[1]
        logit_rho_post   = logit(rho) \
                           + self.B / 2.0 * (logdet_post_cov - logdet_prior_cov) \
                           + 0.5 * post_mu.dot(post_prec).dot(post_mu) \
                           - 0.5 * mu_w.dot(np.linalg.solve(Sigma_w, mu_w))

        rho_post = logistic(logit_rho_post)

        # Sample the binary indicator of an edge
        self.A[n_pre, n_post] = np.random.rand() < rho_post

    def _resample_W(self, n_pre, n_post, stats):
        """
        Resample the weight of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        post_mu, post_cov, post_prec = stats

        self.W[n_pre, n_post, :] = np.random.multivariate_normal(post_mu, post_cov)

