"""
Weight models
"""
import numpy as np
from scipy.linalg.lapack import dpotrs

from pyglm.abstractions import Component
from pyglm.internals.distributions import Bernoulli, Gaussian, TruncatedScalarGaussian
from pyglm.utils.utils import logistic, logit, normal_cdf, sample_truncnorm

from pyglm.utils.profiling import line_profiled
PROFILING = False

class NoWeights(Component):
    def __init__(self, population):
        self.population = population

        # Hard code the parameters to zero (no connections)
        self.A = np.zeros((self.N, self.N))
        self.W = np.zeros((self.N, self.N, self.B))

        # TODO: Fix hack
        self.E_A = self.A
        self.E_W = self.W
        self.E_WWT = np.zeros((self.N, self.N, self.B, self.B))

    @property
    def N(self):
        return self.population.N

    @property
    def B(self):
        return self.population.B

    @property
    def W_effective(self):
        return self.W

    def initialize_with_standard_model(self, standard_model):
        pass

    def resample(self, augmented_data):
        pass

    def meanfieldupdate(self, augmented_data):
        pass

    def get_vlb(self, augmented_data):
        return 0

    def resample_from_mf(self, augmented_data):
        pass

    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        pass


class _SpikeAndSlabGaussianWeightsBase(Component):
    def __init__(self, population):
        self.population = population

        # Initialize the parameters
        self._A = np.zeros((self.N, self.N))
        self._W = np.zeros((self.N, self.N, self.B))

    @property
    def N(self):
        return self.population.N

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        self._W = value

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

    def initialize_from_prior(self):
        P = self.network.adjacency.P
        Mu = self.network.weights.Mu
        Sigma = self.network.weights.Sigma

        self.A = np.random.rand(self.N, self.N) < P

        W = np.zeros((self.N, self.N, self.B))
        for n1 in xrange(self.N):
            for n2 in xrange(self.N):
                W[n1,n2] = np.random.multivariate_normal(Mu[n1,n2], Sigma[n1,n2])
        self.W = W

    def initialize_with_standard_model(self, standard_model, threshold=75):
        """
        Initialize with the weights from a standard model
        :param standard_model:
        :param threshold:      percentile [0,100] of minimum weight
        :return:
        """
        W_std = standard_model.W

        # Make sure it is the correct shape before copying
        assert W_std.shape == (self.N, self.N, self.B)
        self.W = W_std.copy()

        # Keep all the connections
        if threshold is None:
            self.A = np.ones((self.N, self.N))
        else:
            # Only keep the weights that exceed the threshold
            assert threshold >= 0 and threshold <= 100
            W_cutoff = np.percentile(abs(self.W.sum(2)), threshold)
            self.A = abs(self.W.sum(2)) > W_cutoff

    def log_prior(self):
        lprior = 0
        P = self.network.adjacency.P
        Mu = self.network.weights.Mu
        Sigma = self.network.weights.Sigma

        # Log prob of connections
        lprior += Bernoulli(P).log_probability(self.A).sum()

        # Log prob of weights
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                lprior += self.A[n_pre,n_post] * \
                          (Gaussian(Mu[n_pre,n_post,:],
                                    Sigma[n_pre,n_post,:,:])
                           .log_probability(self.W[n_pre,n_post])).sum()
        return lprior


class _GibbsSpikeAndSlabGaussianWeights(_SpikeAndSlabGaussianWeightsBase):
    def __init__(self, population):
        super(_GibbsSpikeAndSlabGaussianWeights, self).__init__(population)

        self.resample()

    def resample(self, augmented_data=[]):

        # TODO: Handle lists of data
        if not isinstance(augmented_data, list):
            augmented_data = [augmented_data]

        # Precompute Psi. We can update it as we add and remove edges
        Psis = [self.activation.compute_psi(data) for data in augmented_data]

        #  TODO: We can parallelize over n_post
        for n_post in xrange(self.N):
            # Randomly permute the order in which we resample presynaptic weights
            perm = np.random.permutation(self.N)
            for n_pre in perm:
                # Get the filtered spike trains associated with this synapse
                F_pres = [data["F"][:,n_pre,:] for data in augmented_data]

                # Compute the activation from other neurons
                if self.A[n_pre, n_post]:
                    for Psi, F_pre in zip(Psis, F_pres):
                        Psi[:,n_post] -= F_pre.dot(self.W[n_pre, n_post,:])

                psi_others = [Psi[:,n_post] for Psi in Psis]


                # Compute the sufficient statistics for this synapse
                suff_stats = self._get_sufficient_statistics(augmented_data,
                                                             n_pre, n_post,
                                                             psi_others)
                post_stats = self._posterior_statistics(n_pre, n_post,
                                                        suff_stats)


                # Sample the spike variable
                self._resample_A(n_pre, n_post, post_stats)

                # Sample the slab variable
                if self.A[n_pre, n_post]:
                    self._resample_W(n_pre, n_post, post_stats)
                else:
                    self.W[n_pre, n_post,:] = 0.0

                # Update Psi to account for new weight
                for F_pre, Psi, psi_other in zip(F_pres, Psis, psi_others):
                    Psi[:,n_post] = psi_other
                    if self.A[n_pre, n_post]:
                        Psi[:,n_post] += F_pre.dot(self.W[n_pre, n_post,:])

    def _get_sufficient_statistics(self, augmented_data, n_pre, n_post, psi_others):
        """
        Get the sufficient statistics for this synapse.
        """
        lkhd_prec           = 0
        lkhd_mean_dot_prec  = 0

        # Compute the sufficient statistics of the likelihood
        for data, psi_other in zip(augmented_data, psi_others):
            lkhd_prec           += self.activation.precision(data, synapse=(n_pre,n_post))
            lkhd_mean_dot_prec  += self.activation.mean_dot_precision(data,
                                                                      synapse=(n_pre,n_post),
                                                                      psi_other=psi_other)

        return lkhd_prec, lkhd_mean_dot_prec

    def _posterior_statistics(self, n_pre, n_post, stats):
        lkhd_prec, lkhd_mean_dot_prec = stats

        mu_w                = self.network.weights.Mu[n_pre, n_post, :]
        Sigma_w             = self.network.weights.Sigma[n_pre, n_post, :, :]

        prior_prec          = np.linalg.inv(Sigma_w)
        prior_mean_dot_prec = mu_w.dot(prior_prec)

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
        mu_w                         = self.network.weights.Mu[n_pre, n_post, :]
        Sigma_w                      = self.network.weights.Sigma[n_pre, n_post, :, :]
        post_mu, post_cov, post_prec = stats
        rho                          = self.network.adjacency.P[n_pre, n_post]

        # Compute the log odds ratio
        logdet_prior_cov = np.linalg.slogdet(Sigma_w)[1]
        logdet_post_cov  = np.linalg.slogdet(post_cov)[1]
        logit_rho_post   = logit(rho) \
                           + 0.5 * (logdet_post_cov - logdet_prior_cov) \
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


class _CollapsedGibbsSpikeAndSlabGaussianWeights(_SpikeAndSlabGaussianWeightsBase):
    def __init__(self, population):
        super(_CollapsedGibbsSpikeAndSlabGaussianWeights, self).__init__(population)

        self.collapsed_resample()

    @property
    def bias_model(self):
        return self.population.bias_model

    def collapsed_resample(self, augmented_data=[]):
        if not isinstance(augmented_data, list):
            augmented_data = [augmented_data]

        self._serial_collapsed_resample(augmented_data)
        # self._parallel_collapsed_resample(augmented_data)

    def _serial_collapsed_resample(self, augmented_data):
        P = self.network.adjacency.P
        for n in xrange(self.N):
            # Compute the prior and posterior sufficient statistics of W
            J_prior, h_prior = self._prior_sufficient_statistics(n)
            J_lkhd, h_lkhd = self._lkhd_sufficient_statistics(n, augmented_data)
            J_post = J_prior + J_lkhd
            h_post = h_prior + h_lkhd

            self._collapsed_resample_A(n, P, J_prior, h_prior, J_post, h_post)
            self._collapsed_resample_W_b(n, J_post, h_post)

    def _parallel_collapsed_resample(self, augmented_data=[]):
        """
        Resample A and W. This  version uses joblib to parallelize
        over columns of the network.
        """
        if not isinstance(augmented_data, list):
            augmented_data = [augmented_data]

        # Use the module trick to avoid copying globals
        import pyglm.internals.parallel as par
        par.augmented_data = augmented_data
        par.weight_model = self
        par.bias_model = self.bias_model
        par.network = self.network
        par.P = self.network.adjacency.P
        par.N = self.N
        par.B = self.B

        # We can naively parallelize over receiving neurons, k2
        # To avoid serializing and copying the data object, we
        # manually extract the required arrays Sk, Fk, etc.
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(par._parallel_collapsed_resample_column)(n) for n in range(self.N))

        # Parse the results
        b = np.array([r[0] for r in results])
        A = np.array([r[1] for r in results]).T
        W = np.array([r[2] for r in results]).transpose(1,0,2)

        self.bias_model.b = b
        self.A = A
        self.W = W

    def _prior_sufficient_statistics(self, n):
        mu_b    = self.bias_model.mu_0
        sigma_b = self.bias_model.sigma_0

        mu_w    = self.network.weights.Mu[:, n, :]
        Sigma_w = self.network.weights.Sigma[:, n, :, :]

        # Create a vector mean and a block diagonal covariance matrix
        from scipy.linalg import block_diag
        mu_full = np.concatenate(([mu_b], mu_w.ravel()))
        Sigma_full = block_diag([[sigma_b]], *Sigma_w)

        # Make sure they are the right size
        assert mu_full.ndim == 1
        K = mu_full.shape[0]
        assert Sigma_full.shape == (K,K)

        # Compute the information form
        J_prior = np.linalg.inv(Sigma_full)
        h_prior = J_prior.dot(mu_full)

        return J_prior, h_prior

    def _lkhd_sufficient_statistics(self, n, augmented_data):
        """
        Compute the full likelihood sufficient statistics as if all connections
         were present.
        """
        # Compute the sufficient statistics of the likelihood
        # These will be the same shape as those of the prior
        D = 1 + self.N * self.B
        J_lkhd = np.zeros((D, D))
        h_lkhd = np.zeros(D)

        # Compute the posterior sufficient statistics
        for data in augmented_data:
            T = data["T"]
            omega = data["omega"]
            kappa = data["kappa"]

            F_flat = data["F_full"]
            assert F_flat.shape == (T,self.N*self.B+1)

            # The likelihood terms will be dense
            # h_lkhd is a 1xT vector times a T x NB matrix
            # We will only need a subset of the resulting NB vector
            J_lkhd += (F_flat * omega[:,n][:,None]).T.dot(F_flat)
            h_lkhd  += kappa[:,n].dot(F_flat)

        return J_lkhd, h_lkhd

    def _marginal_likelihood(self, n, J_prior, h_prior, J_post, h_post):
        """
        Compute the marginal likelihood as the ratio of log normalizers
        """
        Aeff = np.concatenate(([1], np.repeat(self.A[:,n], self.B))).astype(np.bool)

        # Extract the entries for which A=1
        J0 = J_prior[np.ix_(Aeff, Aeff)]
        h0 = h_prior[Aeff]
        Jp = J_post[np.ix_(Aeff, Aeff)]
        hp = h_post[Aeff]

        # This relates to the mean/covariance parameterization as follows
        # log |C| = log |J^{-1}| = -log |J|
        # and
        # mu^T C^{-1} mu = mu^T h
        #                = mu C^{-1} C h
        #                = h^T C h
        #                = h^T J^{-1} h
        # ml = 0
        # ml -= 0.5*np.linalg.slogdet(Jp)[1]
        # ml += 0.5*np.linalg.slogdet(J0)[1]
        # ml += 0.5*hp.dot(np.linalg.solve(Jp, hp))
        # ml -= 0.5*h0.T.dot(np.linalg.solve(J0, h0))

        # Now compute it even faster using the Cholesky!
        L0 = np.linalg.cholesky(J0)
        Lp = np.linalg.cholesky(Jp)

        ml = 0
        ml -= np.sum(np.log(np.diag(Lp)))
        ml += np.sum(np.log(np.diag(L0)))
        ml += 0.5*hp.T.dot(dpotrs(Lp, hp, lower=True)[0])
        ml -= 0.5*h0.T.dot(dpotrs(L0, h0, lower=True)[0])

        return ml

    def _marginal_likelihood_with_block_update(self,
        n, J0_mat, h_prior, Jp_mat, h_post):
        """
        Compute the marginal likelihood as the ratio of log normalizers
        """
        Aeff = np.concatenate(([1], np.repeat(self.A[:,n], self.B))).astype(np.bool)
        Ainds = np.where(Aeff)[0]

        # Extract the entries for which A=1
        h0 = h_prior[Aeff]
        hp = h_post[Aeff]

        # This relates to the mean/covariance parameterization as follows
        # log |C| = log |J^{-1}| = -log |J|
        # and
        # mu^T C^{-1} mu = mu^T h
        #                = mu C^{-1} C h
        #                = h^T C h
        #                = h^T J^{-1} h
        J0inds, J0inv, J0ldet = J0_mat.compute_submatrix_inverse(Ainds)
        Jpinds, Jpinv, Jpldet = Jp_mat.compute_submatrix_inverse(Ainds)

        ml = 0
        ml -= 0.5*Jpldet
        ml += 0.5*J0ldet
        ml += 0.5*hp.dot(Jpinv.dot(hp))
        ml -= 0.5*h0.T.dot(J0inv.dot(h0))

        return ml, (J0inds, J0inv, J0ldet), (Jpinds, Jpinv, Jpldet)


    def _collapsed_resample_A_slow(self, n, P, J_prior, h_prior, J_post, h_post):
        """
        Resample the presence or absence of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        # Sample A[m,n] -- the presence or absence of a connections from
        # presynaptic neuron m -- given the rest of the incoming A[:,n]
        perm = np.random.permutation(self.N)
        for m in perm:
            # Compute the marginal prob with and without A[m,n]
            lps = np.zeros(2)
            for v in [0,1]:
                self.A[m,n] = v

                # Now compute the marginal likelihood using J and h
                lps[v] = self._marginal_likelihood(n,
                                                   J_prior, h_prior,
                                                   J_post, h_post)
                lps[v] += v * np.log(P[m,n]) + (1-v) * np.log(1-P[m,n])

            # Sample from the marginal probability
            max_lps = max(lps[0], lps[1])
            se_lps = np.sum(np.exp(lps-max_lps))
            lse_lps = np.log(se_lps) + max_lps
            ps = np.exp(lps - lse_lps)
            # ps_old = np.exp(lps - logsumexp(lps))
            # assert np.allclose(ps, ps_old)
            # assert np.allclose(ps.sum(), 1.0)
            self.A[m,n] = np.random.rand() < ps[1]

    def _collapsed_resample_A_with_block_update(self, n, P, J_prior, h_prior, J_post, h_post):
        """
        Resample the presence or absence of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        # First sample A[:,n] -- the presence or absence of a connections from
        # presynaptic neurons
        perm = np.random.permutation(self.N)

        # Initialize a big matrix
        from pyglm.utils.fastinv import BigInvertibleMatrix
        J0mat = BigInvertibleMatrix(J_prior)
        Jpmat = BigInvertibleMatrix(J_post)
        ml_prev, J0prms, Jpprms = \
            self._marginal_likelihood_with_block_update(n, J0mat, h_prior, Jpmat, h_post)
        J0mat.update(*J0prms)
        Jpmat.update(*Jpprms)

        for m in perm:

            # Compute the marginal prob with and without A[m,n]
            lps = np.zeros(2)

            # We already have the marginal likelihood for the current value of A
            # We just need to add the prior
            v_prev = self.A[m,n]
            lps[v_prev] += ml_prev
            lps[v_prev] += v_prev * np.log(P[m,n]) + (1-v_prev) * np.log(1-P[m,n])

            # Now compute the posterior stats for 1-v
            v_new = 1 - v_prev
            self.A[m,n] = v_new
            ml_new, J0prms, Jpprms = \
                self._marginal_likelihood_with_block_update(n, J0mat, h_prior, Jpmat, h_post)

            lps[v_new] += ml_new
            lps[v_new] += v_new * np.log(P[m,n]) + (1-v_new) * np.log(1-P[m,n])

            # Sample from the marginal probability
            max_lps = max(lps[0], lps[1])
            se_lps = np.sum(np.exp(lps-max_lps))
            lse_lps = np.log(se_lps) + max_lps
            ps = np.exp(lps - lse_lps)

            # ps = np.exp(lps - logsumexp(lps))
            # assert np.allclose(ps.sum(), 1.0)
            v_smpl = np.random.rand() < ps[1]
            self.A[m,n] = v_smpl

            # Cache the posterior stats and update the matrix objects
            if v_smpl != v_prev:
                ml_prev = ml_new
                J0mat.update(*J0prms)
                Jpmat.update(*Jpprms)


    def _collapsed_resample_A(self, n, P, J_prior, h_prior, J_post, h_post):
        """
        Resample the presence or absence of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        # First sample A[:,n] -- the presence or absence of a connections from
        # presynaptic neurons
        perm = np.random.permutation(self.N)

        ml_prev = self._marginal_likelihood(n, J_prior, h_prior, J_post, h_post)

        for m in perm:

            # Compute the marginal prob with and without A[m,n]
            lps = np.zeros(2)

            # We already have the marginal likelihood for the current value of A
            # We just need to add the prior
            v_prev = self.A[m,n]
            lps[v_prev] += ml_prev
            lps[v_prev] += v_prev * np.log(P[m,n]) + (1-v_prev) * np.log(1-P[m,n])

            # Now compute the posterior stats for 1-v
            v_new = 1 - v_prev
            self.A[m,n] = v_new

            ml_new = self._marginal_likelihood(n, J_prior, h_prior, J_post, h_post)

            lps[v_new] += ml_new
            lps[v_new] += v_new * np.log(P[m,n]) + (1-v_new) * np.log(1-P[m,n])

            # Sample from the marginal probability
            max_lps = max(lps[0], lps[1])
            se_lps = np.sum(np.exp(lps-max_lps))
            lse_lps = np.log(se_lps) + max_lps
            ps = np.exp(lps - lse_lps)

            # ps = np.exp(lps - logsumexp(lps))
            # assert np.allclose(ps.sum(), 1.0)
            v_smpl = np.random.rand() < ps[1]
            self.A[m,n] = v_smpl

            # Cache the posterior stats and update the matrix objects
            if v_smpl != v_prev:
                ml_prev = ml_new


    def _collapsed_resample_W_b(self, n, J_post, h_post):
        """
        Resample the weight of a connection (synapse)
        """
        Aeff = np.concatenate(([1], np.repeat(self.A[:,n], self.B))).astype(np.bool)
        Jp = J_post[np.ix_(Aeff, Aeff)]
        hp = h_post[Aeff]

        # Sample in mean and covariance (standard) form
        # mup = np.linalg.solve(Jp, hp)
        # Sigp = np.linalg.inv(Jp)
        # Wbn = np.random.multivariate_normal(mup, Sigp)

        # Sample in information form
        from pyglm.utils.utils import sample_gaussian
        Wbn = sample_gaussian(J=Jp, h=hp)

        # Set bias and weights
        self.bias_model.b[n] = Wbn[0]
        self.W[:,n,:] = 0
        self.W[self.A[:,n].astype(np.bool),n,:] = Wbn[1:].reshape((-1,self.B))


class _MeanFieldSpikeAndSlabGaussianWeights(_SpikeAndSlabGaussianWeightsBase):
    def __init__(self, population):
        super(_MeanFieldSpikeAndSlabGaussianWeights, self).__init__(population)

        # Initialize the mean field variational parameters
        self.mf_p     = 0.5 * np.ones((self.N, self.N))
        self.mf_mu    = np.zeros((self.N, self.N, self.B))
        self.mf_Sigma = np.tile(np.eye(self.B)[None, None, :, :], (self.N, self.N, 1, 1))

    @property
    def E_A(self):
        return self.mf_p

    @property
    def E_W(self):
        return self.mf_expected_w_given_A(1)

    @property
    def E_WWT(self):
        return self.mf_expected_wwT_given_A(1)

    def initialize_with_standard_model(self, standard_model, threshold=98):
        """
        Initialize with the weights from a standard model
        :param standard_model:
        :return:
        """
        super(_MeanFieldSpikeAndSlabGaussianWeights, self).\
            initialize_with_standard_model(standard_model)

        W_std = standard_model.W
        # Make sure it is the correct shape before copying
        assert W_std.shape == (self.N, self.N, self.B)

        # Keep all the connections
        if threshold is None:
            self.A = np.ones((self.N, self.N))
        else:
            # Only keep the weights that exceed the threshold
            assert threshold >= 0 and threshold <= 100
            W_cutoff = np.percentile(abs(self.W.sum(2)), threshold)
            mf_A = abs(self.W.sum(2)) > W_cutoff
            self.mf_p = 0.9 * mf_A + 0.1 * (1-mf_A)

        # self.mf_p = 0.9 * np.ones((self.N, self.N))
        self.mf_mu = W_std.copy()
        # self.mf_Sigma = np.tile(self.network.weight_dist.sigma_0 * np.eye(self.B)[None, None, :, :], (self.N, self.N, 1, 1))
        self.mf_Sigma = np.tile(1e-5 * np.eye(self.B)[None, None, :, :], (self.N, self.N, 1, 1))

    def old_meanfieldupdate(self, augmented_data):

        # Get network expectations
        E_ln_rho       = self.network.mf_expected_log_p()
        E_ln_notrho    = self.network.mf_expected_log_notp()
        E_mu           = self.network.mf_expected_mu()
        E_Sigma_inv    = self.network.mf_expected_Sigma_inv()
        E_logdet_Sigma = self.network.mf_expected_logdet_Sigma()

        E_net = E_ln_rho, E_ln_notrho, E_mu, E_Sigma_inv, E_logdet_Sigma


        #  TODO: We can parallelize over n_post
        for n_post in xrange(self.N):
            # Randomly permute the order in which we resample presynaptic weights
            perm = np.random.permutation(self.N)
            for n_pre in perm:
                stats = self._get_expected_sufficient_statistics(augmented_data, E_net, n_pre, n_post)

                # Mean field update the slab variable
                self._meanfieldupdate_W(n_pre, n_post, stats)

                # Mean field update the spike variable
                self._meanfieldupdate_A(n_pre, n_post, stats, E_net)

    def meanfieldupdate(self, augmented_data):

        # Get network expectations
        E_ln_rho       = self.network.mf_expected_log_p()
        E_ln_notrho    = self.network.mf_expected_log_notp()
        E_mu           = self.network.mf_expected_mu()
        E_Sigma_inv    = self.network.mf_expected_Sigma_inv()
        E_logdet_Sigma = self.network.mf_expected_logdet_Sigma()

        E_net = E_ln_rho, E_ln_notrho, E_mu, E_Sigma_inv, E_logdet_Sigma

        # Precompute E[\psi]. We can update it as we add and remove edges
        E_psi = self.activation.mf_expected_activation(augmented_data)

        for n_post in xrange(self.N):
            # Randomly permute the order in which we resample presynaptic weights
            perm = np.random.permutation(self.N)
            for n_pre in perm:
                # Get the filtered spike trains associated with this synapse
                F_pre = augmented_data["F"][:,n_pre,:]

                # Compute the expected activation from other neurons
                E_W = self.mf_p[n_pre, n_post] * self.mf_mu[n_pre, n_post,:]
                E_psi[:,n_post] -= F_pre.dot(E_W)
                E_psi_other = E_psi[:,n_post]

                stats = self._get_expected_sufficient_statistics(
                    augmented_data, E_net,
                    n_pre, n_post,
                    E_psi_other=E_psi_other)

                # Mean field update the slab variable
                self._meanfieldupdate_W(n_pre, n_post, stats)

                # Mean field update the spike variable
                self._meanfieldupdate_A(n_pre, n_post, stats, E_net)

                # Update Psi to account for new weight
                E_psi[:,n_post] = E_psi_other
                E_psi[:,n_post] += F_pre.dot(self.mf_p[n_pre, n_post] *
                                             self.mf_mu[n_pre, n_post,:])

    def _get_expected_sufficient_statistics(
            self, augmented_data, E_net,
            n_pre, n_post,
            E_psi_other=None,
            minibatchfrac=1.0):
        """
        Get the expected sufficient statistics for this synapse.
        """
        E_ln_rho, E_ln_notrho, E_mu, E_Sigma_inv, E_logdet_Sigma = E_net
        mu_w                = E_mu[n_pre, n_post, :]
        prec_w              = E_Sigma_inv[n_pre, n_post, :, :]

        prior_prec          = np.linalg.inv(prec_w)
        prior_mean_dot_prec = mu_w.dot(prior_prec)

        # Compute the posterior parameters
        if augmented_data is not None:
            lkhd_prec           = self.activation.mf_precision(augmented_data, synapse=(n_pre,n_post))
            lkhd_mean_dot_prec  = self.activation.mf_mean_dot_precision(augmented_data,
                                                                        synapse=(n_pre,n_post),
                                                                        E_psi_other=E_psi_other)
        else:
            lkhd_prec           = 0
            lkhd_mean_dot_prec  = 0

        post_prec           = prior_prec + lkhd_prec  / minibatchfrac
        post_cov            = np.linalg.inv(post_prec)
        post_mu             = (prior_mean_dot_prec + lkhd_mean_dot_prec  / minibatchfrac ).dot(post_cov)
        post_mu             = post_mu.ravel()

        return post_mu, post_cov, post_prec

    def _meanfieldupdate_A(self, n_pre, n_post, stats, E_net, stepsize=1.0):
        """
        Mean field update the presence or absence of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        # TODO: A joint factor for mu and Sigma could yield E_mu_dot_Sigma under the priro
        mf_post_mu, mf_post_cov, mf_post_prec = stats
        E_ln_rho, E_ln_notrho, E_mu, E_Sigma_inv, E_logdet_Sigma = E_net

        E_ln_rho       = E_ln_rho[n_pre, n_post]
        E_ln_notrho    = E_ln_notrho[n_pre, n_post]
        E_mu           = E_mu[n_pre, n_post,:]
        E_Sigma_inv    = E_Sigma_inv[n_pre, n_post,:,:]
        E_logdet_Sigma = E_logdet_Sigma[n_pre, n_post]

        # Compute the log odds ratio
        logdet_prior_cov = E_logdet_Sigma
        logdet_post_cov  = np.linalg.slogdet(mf_post_cov)[1]
        logit_rho_post   = E_ln_rho - E_ln_notrho \
                           + 0.5 * (logdet_post_cov - logdet_prior_cov) \
                           + 0.5 * mf_post_mu.dot(mf_post_prec).dot(mf_post_mu) \
                           - 0.5 * E_mu.dot(E_Sigma_inv.dot(E_mu))

        rho_post = logistic(logit_rho_post)

        # Mean field update the binary indicator of an edge
        self.mf_p[n_pre, n_post] = (1.0 - stepsize) * self.mf_p[n_pre, n_post] \
                                   + stepsize * rho_post

        # logit_rho_post = (1-stepsize) * logit(self.mf_p[n_pre, n_post]) + \
        #                  stepsize * logit_rho_post
        # self.mf_p[n_pre, n_post] = logistic(logit_rho_post)

    def _meanfieldupdate_W(self, n_pre, n_post, stats, stepsize=1.0):
        """
        Resample the weight of a connection (synapse)
        :param n_pre:
        :param n_post:
        :param stats:
        :return:
        """
        mf_post_mu, mf_post_cov, _ = stats

        self.mf_mu[n_pre, n_post, :]       = (1-stepsize) * self.mf_mu[n_pre, n_post, :] \
                                             + stepsize * mf_post_mu
        self.mf_Sigma[n_pre, n_post, :, :] = (1-stepsize) * self.mf_Sigma[n_pre, n_post, :, :] \
                                             + stepsize * mf_post_cov

    def mf_expected_w_given_A(self, A):
        return A * self.mf_mu

    def mf_expected_wwT_given_A(self, A):
        if A == 1:
            # mumuT_dbg = np.zeros((self.N, self.N, self.B, self.B))
            # for n_pre in xrange(self.N):
            #     for n_post in xrange(self.N):
            #         mumuT_dbg[n_pre, n_post, :, :] = np.outer(self.mf_mu[n_pre, n_post, :],
            #                                               self.mf_mu[n_pre, n_post, :])

            # TODO: Compute with einsum instead
            mumuT = np.einsum("ijk,ijl->ijkl", self.mf_mu, self.mf_mu)
            # assert np.allclose(mumuT, mumuT_dbg)

            return self.mf_Sigma + mumuT
        else:
            return np.zeros((self.N, self.N, self.B, self.B))

    def mf_expected_wwT(self):
        return self.mf_p[:,:,None,None] * self.mf_expected_wwT_given_A(1)

    def mf_expected_W(self):
        return self.mf_p[:,:,None] * self.mf_mu

    # def mf_expected_wwT(self, n_pre, n_post):
    #     """
    #     E[ww^T] = E_{A}[ E_{W|A}[ww^T | A] ]
    #             = rho * E[ww^T | A=1] + (1-rho) * 0
    #     :return:
    #     """
    #     mumuT = np.outer(self.mf_mu[n_pre, n_post, :], self.mf_mu[n_pre, n_post, :])
    #     return self.mf_p[n_pre, n_post] * (self.mf_Sigma[n_pre, n_post, :, :] + mumuT)

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

    def mf_mode(self):
        for n_pre in np.arange(self.N):
            for n_post in np.arange(self.N):
                self.A[n_pre, n_post] = 1
                self.W[n_pre, n_post, :] = self.mf_p[n_pre, n_post] * self.mf_mu[n_pre, n_post, :]

    ### SVI
    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        # Get network expectations
        E_ln_rho       = self.network.mf_expected_log_p()
        E_ln_notrho    = self.network.mf_expected_log_notp()
        E_mu           = self.network.mf_expected_mu()
        E_Sigma_inv    = self.network.mf_expected_Sigma_inv()
        E_logdet_Sigma = self.network.mf_expected_logdet_Sigma()

        E_net = E_ln_rho, E_ln_notrho, E_mu, E_Sigma_inv, E_logdet_Sigma

        # Precompute E[\psi]. We can update it as we add and remove edges
        E_psi = self.activation.mf_expected_activation(augmented_data)

        for n_post in xrange(self.N):
            # Randomly permute the order in which we resample presynaptic weights
            perm = np.random.permutation(self.N)
            for n_pre in perm:
                # Get the filtered spike trains associated with this synapse
                F_pre = augmented_data["F"][:,n_pre,:]

                # Compute the expected activation from other neurons
                E_W = self.mf_p[n_pre, n_post] * self.mf_mu[n_pre, n_post,:]
                E_psi[:,n_post] -= F_pre.dot(E_W)
                E_psi_other = E_psi[:,n_post]

                stats = self._get_expected_sufficient_statistics(
                    augmented_data, E_net,
                    n_pre, n_post,
                    E_psi_other=E_psi_other,
                    minibatchfrac=minibatchfrac)

                # Mean field update the slab variable
                self._meanfieldupdate_W(n_pre, n_post, stats,
                                        stepsize=stepsize)

                # Mean field update the spike variable
                self._meanfieldupdate_A(n_pre, n_post, stats, E_net,
                                        stepsize=stepsize)

                # Update Psi to account for new weight
                E_psi[:,n_post] = E_psi_other
                E_psi[:,n_post] += F_pre.dot(self.mf_p[n_pre, n_post] *
                                             self.mf_mu[n_pre, n_post,:])

class SpikeAndSlabGaussianWeights(_GibbsSpikeAndSlabGaussianWeights,
                                  _CollapsedGibbsSpikeAndSlabGaussianWeights,
                                  _MeanFieldSpikeAndSlabGaussianWeights):
    pass


class SpikeAndSlabTruncatedGaussianWeights(_SpikeAndSlabGaussianWeightsBase):
    """
    Base class for truncated spike and slab Gaussian weights
    """
    def __init__(self, population, lb=-np.Inf, ub=np.Inf):
        self.lb = lb
        self.ub = ub

        super(SpikeAndSlabTruncatedGaussianWeights, self).\
            __init__(population)

        self.resample()


    def initialize_with_standard_model(self, standard_model, threshold=75):
        """
        Initialize with the weights from a standard model
        :param standard_model:
        :param threshold:      percentile [0,100] of minimum weight
        :return:
        """
        W_std = standard_model.W

        # Make sure it is the correct shape before copying
        assert W_std.shape == (self.N, self.N, self.B)

        # Clip the standard weights to the truncated range
        self.W = np.clip(W_std.copy(), self.lb, self.ub)

        # Keep all the connections
        if threshold is None:
            self.A = np.ones((self.N, self.N))
        else:
            # Only keep the weights that exceed the threshold
            assert threshold >= 0 and threshold <= 100
            W_cutoff = np.percentile(abs(self.W.sum(2)), threshold)
            self.A = abs(self.W.sum(2)) > W_cutoff

    def log_prior(self):
        lprior = 0
        P = self.network.adjacency.P
        Mu = self.network.weights.Mu
        Sigma = self.network.weights.Sigma

        # Log prob of connections
        lprior += Bernoulli(P).log_probability(self.A).sum()

        # Log prob of weights
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                for b in xrange(self.B):
                    lprior += self.A[n_pre,n_post] * \
                              (TruncatedScalarGaussian(Mu[n_pre,n_post,b],
                                                       Sigma[n_pre,n_post,b,b],
                                                       self.lb, self.ub)
                               .log_probability(self.W[n_pre,n_post,b]))

                    if not np.isfinite(lprior):
                        import pdb; pdb.set_trace()

        return lprior


    def resample(self, augmented_data=[]):
        # Handle lists of data
        if not isinstance(augmented_data, list):
            augmented_data = [augmented_data]

        # Precompute Psi. We can update it as we add and remove edges
        Psis = [self.activation.compute_psi(data) for data in augmented_data]

        #  TODO: We can parallelize over n_post
        for n_post in xrange(self.N):

            # Randomly permute the order in which we resample presynaptic weights
            perm = np.random.permutation(self.N)
            for n_pre in perm:
                # Get the filtered spike trains associated with this synapse
                F_pres = [data["F"][:,n_pre,:] for data in augmented_data]

                # Compute the activation from other neurons
                if self.A[n_pre, n_post]:
                    for Psi, F_pre in zip(Psis, F_pres):
                        Psi[:,n_post] -= F_pre.dot(self.W[n_pre, n_post,:])

                psi_others = [Psi[:,n_post] for Psi in Psis]


                # Compute the sufficient statistics for this synapse
                suff_stats = self._get_sufficient_statistics(augmented_data,
                                                             n_pre, n_post,
                                                             psi_others)
                post_stats = self._posterior_statistics(n_pre, n_post,
                                                        suff_stats)


                # Sample the spike variable
                self._resample_A(n_pre, n_post, post_stats)

                # Sample the slab variable
                if self.A[n_pre, n_post]:
                    self._resample_W(n_pre, n_post, post_stats)
                else:
                    self.W[n_pre, n_post,:] = 0.0

                # Update Psi to account for new weight
                for F_pre, Psi, psi_other in zip(F_pres, Psis, psi_others):
                    Psi[:,n_post] = psi_other
                    if self.A[n_pre, n_post]:
                        Psi[:,n_post] += F_pre.dot(self.W[n_pre, n_post,:])

    def _get_sufficient_statistics(self, augmented_data, n_pre, n_post, psi_others):
        """
        Get the sufficient statistics for this synapse.
        """
        lkhd_prec           = 0
        lkhd_mean_dot_prec  = 0

        # Compute the sufficient statistics of the likelihood
        for data, psi_other in zip(augmented_data, psi_others):
            lkhd_prec           += self.activation.precision(data, synapse=(n_pre,n_post))
            lkhd_mean_dot_prec  += self.activation.mean_dot_precision(data,
                                                                      synapse=(n_pre,n_post),
                                                                      psi_other=psi_other)

        return lkhd_prec, lkhd_mean_dot_prec

    def _posterior_statistics(self, n_pre, n_post, stats):
        lkhd_prec, lkhd_mean_dot_prec = stats

        mu_w                = self.network.weights.Mu[n_pre, n_post, :]
        Sigma_w             = self.network.weights.Sigma[n_pre, n_post, :, :]

        prior_prec          = np.linalg.inv(Sigma_w)
        prior_mean_dot_prec = mu_w.dot(prior_prec)

        post_prec           = prior_prec + lkhd_prec
        post_cov            = np.linalg.inv(post_prec)

        # TODO: Verify that this is correct.
        # Under an independent, truncated Normal prior for each entry in W,
        # the posterior covariance is simply the diagonal of what it would
        # be under a correlated, multivariate Normal prior
        post_cov = np.diag(np.diag(post_cov))
        post_prec = np.diag(np.diag(post_prec))

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
        prior_mu                         = self.network.weights.Mu[n_pre, n_post, :]
        prior_cov                       = self.network.weights.Sigma[n_pre, n_post, :, :]
        prior_sigmasq = np.diag(prior_cov)

        post_mu, post_cov, post_prec = stats
        post_sigmasq = np.diag(post_cov)
        rho                          = self.network.adjacency.P[n_pre, n_post]

        # Compute the log odds ratio
        logdet_prior_cov = 0.5*np.log(prior_sigmasq).sum()
        logdet_post_cov  = 0.5*np.log(post_sigmasq).sum()
        logit_rho_post   = logit(rho) \
                           + 0.5 * (logdet_post_cov - logdet_prior_cov) \
                           + 0.5 * post_mu.dot(post_prec).dot(post_mu) \
                           - 0.5 * prior_mu.dot(np.linalg.solve(prior_cov, prior_mu))

        # The truncated normal prior introduces another term,
        # the ratio of the normalizers of the truncated distributions
        logit_rho_post += np.log(normal_cdf((self.ub-post_mu) / np.sqrt(post_sigmasq)) -
                                 normal_cdf((self.lb-post_mu) / np.sqrt(post_sigmasq))).sum()

        logit_rho_post -= np.log(normal_cdf((self.ub-prior_mu) / np.sqrt(prior_sigmasq)) -
                                 normal_cdf((self.lb-prior_mu) / np.sqrt(prior_sigmasq))).sum()

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
        post_sigma = np.sqrt(np.diag(post_cov))

        for b in xrange(self.B):
            self.W[n_pre, n_post, b] = sample_truncnorm(post_mu[b], post_sigma[b],
                                                        self.lb, self.ub)

            if not np.isfinite(self.W[n_pre, n_post, b]):
                import pdb; pdb.set_trace()

            if self.W[n_pre, n_post, b] < self.lb or self.W[n_pre, n_post, b] > self.ub:
                import pdb; pdb.set_trace()

    # TODO: Implement mean field
    def E_A(self): raise NotImplementedError()
    def E_W(self): raise NotImplementedError()
    def E_WWT(self): raise NotImplementedError()
    def get_vlb(self, augmented_data): raise NotImplementedError()
    def meanfieldupdate(self, augmented_data): raise NotImplementedError()
    def resample_from_mf(self, augmented_data): raise NotImplementedError()
    def svi_step(self, augmented_data, minibatchfrac, stepsize): raise NotImplementedError()