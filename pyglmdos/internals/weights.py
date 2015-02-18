"""
Weight models
"""
import numpy as np

from pyglmdos.abstractions import Component
from pyglmdos.internals.distributions import Bernoulli, Gaussian

from pyglm.utils.utils import logistic, logit

class _SpikeAndSlabGaussianWeightsBase(Component):
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
    def W_effective(self):
        return self.A[:,:,None] * self.W

    @property
    def network(self):
        return self.population.network

    @property
    def activation(self):
        return self.population.activation_model

    def log_prior(self):
        lprior = 0
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                lprior += Bernoulli(self.network.P[n_pre,n_post]).log_probability(self.A).sum()
                lprior += self.A[n_pre,n_post] * \
                          (Gaussian(self.network.Mu[n_pre,n_post,:],
                                    self.network.Sigma[n_pre,n_post,:,:])
                           .log_probability(self.W[n_pre,n_post])).sum()
        return lprior


class _GibbsSpikeAndSlabGaussianWeights(_SpikeAndSlabGaussianWeightsBase):
    def __init__(self, population):
        super(_GibbsSpikeAndSlabGaussianWeights, self).__init__(population)

        self.resample(None)

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


class _MeanFieldSpikeAndSlabGaussianWeights(_SpikeAndSlabGaussianWeightsBase):
    def __init__(self, population):
        super(_MeanFieldSpikeAndSlabGaussianWeights, self).__init__(population)

        # Initialize the mean field variational parameters
        self.mf_p     = 0.5 * np.ones((self.N, self.N))
        self.mf_mu    = np.zeros((self.N, self.N, self.B))
        self.mf_Sigma = np.tile(np.eye(self.B)[None, None, :, :], (self.N, self.N, 1, 1))

    def meanfieldupdate(self, augmented_data):
        for n_pre in xrange(self.N):
            #  TODO: We can parallelize over n_post
            for n_post in xrange(self.N):
                stats = self._get_expected_sufficient_statistics(augmented_data, n_pre, n_post)

                # Mean field update the slab variable
                self._meanfieldupdate_W(n_pre, n_post, stats)

                # Mean field update the spike variable
                self._meanfieldupdate_A(n_pre, n_post, stats)


    def _get_expected_sufficient_statistics(self, augmented_data, n_pre, n_post):
        """
        Get the expected sufficient statistics for this synapse.
        """
        # TODO: A joint factor for mu and Sigma could yield E_mu_dot_Sigma under the priro
        mu_w                = self.network.mf_Mu[n_pre, n_post, :]
        Sigma_w             = self.network.mf_Sigma[n_pre, n_post, :, :]

        prior_prec          = np.linalg.inv(Sigma_w)
        prior_mean_dot_prec = mu_w.dot(prior_prec)

        # Compute the posterior parameters
        if augmented_data is not None:
            lkhd_prec           = self.activation.mf_precision(augmented_data, synapse=(n_pre,n_post))
            lkhd_mean_dot_prec  = self.activation.mf_mean_dot_precision(augmented_data, synapse=(n_pre,n_post))
        else:
            lkhd_prec           = 0
            lkhd_mean_dot_prec  = 0

        post_prec           = prior_prec + lkhd_prec
        post_cov            = np.linalg.inv(post_prec)
        post_mu             = (prior_mean_dot_prec + lkhd_mean_dot_prec).dot(post_cov)
        post_mu             = post_mu.ravel()

        return post_mu, post_cov, post_prec

    def _meanfieldupdate_A(self, n_pre, n_post, stats):
        """
        Mean field update the presence or absence of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        # TODO: A joint factor for mu and Sigma could yield E_mu_dot_Sigma under the priro
        mf_post_mu, mf_post_cov, mf_post_prec = stats
        # mf_mu_w                         = self.network.mf_Mu[n_pre, n_post, :]
        # mf_Sigma_w                      = self.network.mf_Sigma[n_pre, n_post, :, :]
        # mf_rho                          = self.network.mf_P[n_pre, n_post]
        E_ln_rho       = self.network.mf_expected_log_p()[n_pre, n_post]
        E_ln_notrho    = self.network.mf_expected_log_notp()[n_pre, n_post]
        E_mu           = self.network.mf_expected_mu()[n_pre, n_post, :]
        E_Sigma_inv    = self.network.mf_expected_Sigma_inv()[n_pre, n_post, :, :]
        E_logdet_Sigma = self.network.mf_expected_logdet_Sigma()[n_pre, n_post]

        # Compute the log odds ratio
        logdet_prior_cov = E_logdet_Sigma
        logdet_post_cov  = np.linalg.slogdet(mf_post_cov)[1]
        logit_rho_post   = E_ln_rho - E_ln_notrho \
                           + self.B / 2.0 * (logdet_post_cov - logdet_prior_cov) \
                           + 0.5 * mf_post_mu.dot(mf_post_prec).dot(mf_post_mu) \
                           - 0.5 * E_mu.dot(E_Sigma_inv.dot(E_mu))

        rho_post = logistic(logit_rho_post)

        # Mean field update the binary indicator of an edge
        self.mf_p[n_pre, n_post] = rho_post

    def _meanfieldupdate_W(self, n_pre, n_post, stats):
        """
        Resample the weight of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        mf_post_mu, mf_post_cov, _ = stats

        self.mf_mu[n_pre, n_post, :]       = mf_post_mu
        self.mf_Sigma[n_pre, n_post, :, :] = mf_post_cov

    def mf_expected_w_given_A(self, A):
        return A * self.mf_mu

    def mf_expected_wwT_given_A(self, A):
        if A == 1:
            mumuT = np.zeros((self.N, self.N, self.B, self.B))
            for n_pre in xrange(self.N):
                for n_post in xrange(self.N):
                    mumuT[n_pre, n_post, :, :] = np.outer(self.mf_mu[n_pre, n_post, :],
                                                          self.mf_mu[n_pre, n_post, :])

            # TODO: Compute with einsum instead
            mumuT_einsum = np.einsum("ijk,ijl->ijkl", self.mf_mu, self.mf_mu)
            assert np.allclose(mumuT, mumuT_einsum)

            return self.mf_Sigma + mumuT
        else:
            return np.zeros((self.N, self.N, self.B, self.B))

    def mf_expected_W(self):
        return self.mf_p[:,:,None] * self.mf_mu

    def mf_expected_wwT(self, n_pre, n_post):
        """
        E[ww^T] = E_{A}[ E_{W|A}[ww^T | A] ]
                = rho * E[ww^T | A=1] + (1-rho) * 0
        :return:
        """
        mumuT = np.outer(self.mf_mu[n_pre, n_post, :], self.mf_mu[n_pre, n_post, :])
        return self.mf_p[n_pre, n_post] * (self.mf_Sigma[n_pre, n_post, :, :] + mumuT)

    def get_vlb(self, augmented_data):
        """
        VLB for A and W
        :return:
        """
        vlb = 0

        # Precompute expectations
        E_A            = self.mf_p
        E_notA         = 1.0 - E_A
        E_ln_rho       = self.network.mf_expected_log_p()
        E_ln_notrho    = self.network.mf_expected_log_notp()

        E_W            = self.mf_expected_w_given_A(A=1)
        E_WWT          = self.mf_expected_wwT_given_A(A=1)
        E_mu           = self.network.mf_expected_mu()
        E_mumuT        = self.network.mf_expected_mumuT()
        E_Sigma_inv    = self.network.mf_expected_Sigma_inv()
        E_logdet_Sigma = self.network.mf_expected_logdet_Sigma()

        # E[LN p(A | p)]
        vlb += Bernoulli().negentropy(E_x=E_A, E_notx=E_notA,
                                      E_ln_p=E_ln_rho, E_ln_notp=E_ln_notrho).sum()

        # E[LN q(A | \tilde{p})
        vlb -= Bernoulli(self.mf_p).negentropy().sum()

        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):

                # E[LN p(W | A=1, mu, Sigma)
                vlb += (E_A[n_pre, n_post] * Gaussian().negentropy(E_x=E_W[n_pre, n_post, :],
                                                    E_xxT=E_WWT[n_pre, n_post, :, :],
                                                    E_mu=E_mu[n_pre, n_post, :],
                                                    E_mumuT=E_mumuT[n_pre, n_post, :, :],
                                                    E_Sigma_inv=E_Sigma_inv[n_pre, n_post, :, :],
                                                    E_logdet_Sigma=E_logdet_Sigma[n_pre, n_post])).sum()

                vlb -= (E_A[n_pre, n_post] * Gaussian(self.mf_mu[n_pre, n_post, :],
                                      self.mf_Sigma[n_pre, n_post, :, :]).negentropy()).sum()

        return vlb

    def resample_from_mf(self, augmented_data):
        """
        Resample from the variational distribution
        """
        for n_pre in np.arange(self.N):
            for n_post in np.arange(self.N):
                self.A[n_pre, n_post] = np.random.rand() < self.mf_p[n_pre, n_post]
                self.W[n_pre, n_post, :] = \
                    np.random.multivariate_normal(self.mf_mu[n_pre, n_post, :],
                                                  self.mf_Sigma[n_pre, n_post, :, :])


class SpikeAndSlabGaussianWeights(_GibbsSpikeAndSlabGaussianWeights,
                                  _MeanFieldSpikeAndSlabGaussianWeights):
    pass