import numpy as np
from pyglm.deps.pybasicbayes.abstractions import Collapsed, MeanField
from pyglm.deps.pybasicbayes.distributions import GibbsSampling, GaussianFixed
from pyglm.deps.pybasicbayes.util.stats import sample_discrete_from_log

from pyglm.internals.distributions import Gaussian, Bernoulli

from pyglm.utils.utils import logistic, logit

class GaussianVectorSynapse(GibbsSampling, Collapsed, MeanField):
    def __init__(self,
                 neuron_model,
                 n_pre):

        self.neuron_model = neuron_model
        self.n_pre = n_pre
        self.A = 1
        self.w = self.mu_w

        # Mean field parameters
        self.mf_mu_w = np.copy(self.mu_w)
        self.mf_Sigma_w = np.copy(self.Sigma_w)

        # Create cache for X \dot w
        # self.cache = SimpleCache()
        self.cache = None

        assert self.mu_w.ndim == 1
        self.resample(prior=True) # initialize from prior

    @property
    def weights_prior(self):
        return self.neuron_model.population.\
                     network.weights_prior(self.n_pre,
                                           self.neuron_model.n)

    @property
    def mu_w(self):
        return self.weights_prior.mu

    @property
    def Sigma_w(self):
        return self.weights_prior.sigma

    @property
    def D_in(self):
        # mat = self.w if self.w is not None else self.mu_w
        return self.mu_w.shape[0]

    @property
    def D_out(self):
        # For now, assume we are doing scalar regression
        return 1

    def set_weights(self, value):
        self.w = value
        self.cache = None

    ### getting statistics

    # def _get_statistics(self,data):
    #     if isinstance(data,list):
    #         return sum((self._get_statistics(d) for d in data),self._empty_statistics())
    #     else:
    #         data = data[~np.isnan(data).any(1)]
    #         n, D = data.shape[0], self.D_out
    #
    #         statmat = data.T.dot(data)
    #         xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]
    #
    #         return np.array([yyT, yxT, xxT, n])
    #
    # def _empty_statistics(self):
    #     return np.array([np.zeros((self.D_out, self.D_out)),
    #                      np.zeros((1,self.D_in)),
    #                      np.zeros((self.D_in, self.D_in)),
    #                      0])

    ### distribution

    def log_likelihood(self,xy):
        w, eta, D = self.w, self.neuron_model.eta, self.D_out
        x, y = xy[:,:-D], xy[:,-D:]
        mu_y = x.dot(w.T)

        ll = (-0.5/eta * (y-mu_y)**2).sum()
        ll -= -0.5 * np.log(2*np.pi * eta**2)

        return ll

    def rvs(self,x=None,size=1,return_xy=True):
        w, eta = self.w, self.neuron_model.eta
        x = np.random.normal(size=(size,w.shape[1])) if x is None else x
        y = x.dot(w.T) + np.sqrt(eta) * np.random.normal(size=(x.shape[0],))

        return np.hstack((x,y[:,None])) if return_xy else y

    ### Gibbs sampling
    # def cond_w(self, data=[], stats=None):
    #     ss = self._get_statistics(data) if stats is None else stats
    #     yxT = ss[1]
    #     xxT = ss[2]
    #
    #     # Posterior mean of a Gaussian
    #     Sigma_w_inv = np.linalg.inv(self.Sigma_w)
    #     Sigma_w_post = np.linalg.inv(xxT / self.neuron_model.eta + Sigma_w_inv)
    #     mu_w_post = ((yxT/self.neuron_model.eta + self.mu_w.dot(Sigma_w_inv)).dot(Sigma_w_post)).reshape((self.D_in,))
    #     return GaussianFixed(mu_w_post, Sigma_w_post)
    #
    # def resample(self,data=[],stats=None):
    #     self.A = 1
    #     dist = self.cond_w(data=data, stats=stats)
    #     w = dist.rvs()[0,:]
    #     self.set_weights(w)

    def _get_statistics(self, prior=False):

        prior_prec          = np.linalg.inv(self.Sigma_w)
        prior_mean_dot_prec = self.mu_w.dot(prior_prec)

        if prior:
            post_prec = prior_prec
            post_cov  = np.linalg.inv(post_prec)
            post_mu   = post_cov.dot(prior_mean_dot_prec)
        else:
            # Compute the posterior parameters
            lkhd_prec           = self.neuron_model.activation_lkhd_precision(self.n_pre + 1)
            lkhd_mean_dot_prec  = self.neuron_model.activation_lkhd_mean_dot_precision(self.n_pre + 1)

            post_prec           = prior_prec + lkhd_prec
            post_cov            = np.linalg.inv(post_prec)
            post_mu             = (prior_mean_dot_prec + lkhd_mean_dot_prec).dot(post_cov)
            post_mu             = post_mu.ravel()

        return post_mu, post_cov, post_prec

    def resample(self, prior=False, stats=None):
        """
        Resample the bias given the weights and psi
        :return:
        """
        if stats is None:
            post_mu, post_cov, post_prec = self._get_statistics(prior)
        else:
            post_mu, post_cov, post_prec = stats

        w = np.random.multivariate_normal(post_mu, post_cov)
        self.set_weights(w)

    ### Prediction
    def predict(self, X):
        # if self.cache is not None:
        #     return self.cache
        # else:
        #     y = self._predict_helper(X)
        #     self.cache = y
        #     return y
        return self._predict_helper(X)
        # return self.cache.get(X, self._predict_helper)

    def _predict_helper(self, X):
        w = self.w
        y = X.dot(w.T)
        return y

    ### Collapsed
    def log_marginal_likelihood(self,data):
        # The marginal distribution for multivariate Gaussian mean and
        #  fixed covariance is another multivariate Gaussian.
        if isinstance(data, list):
            out = np.array([self.log_marginal_likelihood(d) for d in data])
        elif isinstance(data, np.ndarray):
            N = data.shape[0]
            X,y = data[:,:-1], data[:,-1]

            # We've implement this with matrix inversion lemma
            # Compute the marginal distribution parameters
            mu_marg = X.dot(self.mu_w.T).reshape((N,))

            from utils.utils import logdet_low_rank2, quad_form_diag_plus_lr2
            # Ainv = 1.0/np.asscalar(self.sigma) * np.eye(N)
            d = np.asscalar(self.neuron_model.eta) * np.ones((N,))
            Ainv = 1.0/d

            # Sig_marg_inv = invert_low_rank(Ainv, X, self.Sigma_A, X.T, diag=True)
            yy = y-mu_marg
            tmp = -1./2. * quad_form_diag_plus_lr2(yy, d, X, self.Sigma_w, X.T)
            out = tmp \
                  - N/2*np.log(2*np.pi) \
                  - 0.5 * logdet_low_rank2(Ainv, X, self.Sigma_w, X.T, diag=True)

            # See test/test_synapses.py for testing code

        else:
            raise Exception("Data must be list of numpy arrays or numpy array")

        return out

    ### MeanField
    def expected_log_likelihood(self, x):
        pass

    def meanfieldupdate(self):
        """
        Perform coordinate ascent on the weights of the Gaussian synapse
        """
        # Extract the covariates and the residuals
        # NOTE: data is assumed to contain the EXPECTED RESIDUAL under the
        # remaining variational distributions

        # # TODO: Use get_weighted_statistics!
        # ss = self._get_statistics(data)
        # yxT = ss[1]
        # xxT = ss[2]
        #
        # # TODO: Use the expected noise variance eta
        # E_eta_inv = self.neuron_model.noise_model.expected_eta_inv()
        # Sigma_w_inv = np.linalg.inv(self.Sigma_w)
        # self.mf_Sigma_w = np.linalg.inv(xxT * E_eta_inv + Sigma_w_inv)
        # self.mf_mu_w = ((yxT * E_eta_inv + self.mu_w.dot(Sigma_w_inv))
        #                 .dot(self.mf_Sigma_w)) \
        #                 .reshape((self.D_in,))

        prior_prec             = np.linalg.inv(self.Sigma_w)
        prior_mean_dot_prec    = self.mu_w.dot(prior_prec)

        # Compute the posterior parameters
        mf_lkhd_prec           = self.neuron_model.mf_activation_lkhd_precision(self.n_pre + 1)
        mf_lkhd_mean_dot_prec  = self.neuron_model.mf_activation_lkhd_mean_dot_precision(self.n_pre + 1)

        mf_post_prec           = prior_prec + mf_lkhd_prec
        mf_post_cov            = np.linalg.inv(mf_post_prec)
        mf_post_mu             = (prior_mean_dot_prec + mf_lkhd_mean_dot_prec).dot(mf_post_cov)
        mf_post_mu             = mf_post_mu.ravel()

        self.mf_mu_w           = mf_post_mu
        self.mf_Sigma_w        = mf_post_cov

    def get_vlb(self):
        vlb = 0

        E_W            = self.mf_expected_w
        E_WWT          = self.mf_expected_wwT
        E_mu           = self.mu_w
        E_mumuT        = self.mu_w.dot(self.mu_w.T)
        E_Sigma_inv    = np.linalg.inv(self.Sigma_w)
        E_logdet_Sigma = np.linalg.slogdet(self.Sigma_w)[1]

        # E[LN p(W | mu, Sigma)
        vlb += Gaussian().negentropy(E_x=E_W, E_xxT=E_WWT,
                                     E_mu=E_mu, E_mumuT=E_mumuT,
                                     E_Sigma_inv=E_Sigma_inv, E_logdet_Sigma=E_logdet_Sigma)

        # E[LN q(W | mu, Sigma)
        vlb -= Gaussian(self.mf_mu_w, self.mf_Sigma_w).negentropy()

    @property
    def mf_expected_w(self):
        return self.mf_mu_w

    @property
    def mf_expected_wwT(self):
        """
        Compute expected w * w^T = Cov(W) + E[w]E[w^T]
        """
        return self.mf_Sigma_w + np.outer(self.mf_expected_w, self.mf_expected_w)

    def mf_predict(self, X):
        w = self.mf_expected_w
        return X.dot(w.T)

    def resample_from_mf(self):
        w = np.random.multivariate_normal(self.mf_mu_w, self.mf_Sigma_w)
        self.set_weights(w)

class SpikeAndSlabGaussianVectorSynapse(GaussianVectorSynapse):
    """
    Combine the Gaussian vector of weights and the Bernoulli indicator variable
    into one object.
    """
    def __init__(self,
                 neuron_model,
                 n_pre):
        super(SpikeAndSlabGaussianVectorSynapse, self).\
            __init__(neuron_model, n_pre)

        # Initialize the mean field parameters
        self.mf_rho = self.rho

        self.resample(prior=True)

    @property
    def rho(self):
        return self.neuron_model.population.network.rho[self.n_pre, self.neuron_model.n]

    def resample(self, prior=False, stats=None):
        """
        Resample the bias given the weights and psi
        :return:
        """
        stats = self._get_statistics(prior=prior)
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

        # Sample the spike variable
        # self.As[m] = log_sum_exp_sample(lp_A)
        self.A = np.random.rand() < rho_post

        # Sample the slab variable
        if self.A:
            super(SpikeAndSlabGaussianVectorSynapse, self).resample(stats=stats)
        else:
            self.w = np.zeros(self.D_in)

    def old_resample(self, data=[]):
        """
        Jointly resample the spike and slab indicator variables and synapse models
        :return:
        """
        rho = self.rho

        # Compute log Pr(A=0|...) and log Pr(A=1|...)
        if len(data) > 0:
            y = data[:,-1]
            if rho > 0.:
                lp_A = np.zeros(2)

                # Residuals are mean zero, variance sigma without a synapse
                mu_0 = np.array([0])
                Sigma_0 = self.neuron_model.eta*np.eye(1)
                lp_A[0] = np.log(1.0-rho) + GaussianFixed(mu_0, Sigma_0)\
                                                .log_likelihood(y).sum()

                # Integrate out the weights to get marginal probability of a synapse
                lp_A[1] = np.log(rho) + self.log_marginal_likelihood(data).sum()

            else:
                lp_A = np.log([1.,0.])

        else:
            # Compute log Pr(A=0|...) and log Pr(A=1|...)
            lp_A = np.zeros(2)
            lp_A[0] = np.log(1.0-rho)
            lp_A[1] = np.log(rho)

        if not np.any(np.isfinite(lp_A)):
            import pdb; pdb.set_trace()

        # Sample the spike variable
        # self.As[m] = log_sum_exp_sample(lp_A)
        self.A = sample_discrete_from_log(lp_A)

        # Sample the slab variable
        if self.A:
            super(SpikeAndSlabGaussianVectorSynapse, self).resample(data)
        else:
            self.w = np.zeros(self.D_in)

    def predict(self, X):
        y = 0
        if self.A:
            w = self.w
            y += X.dot(w.T)

        return y

    ### MeanField
    @property
    def mf_expected_w(self):
        return self.mf_mu_w * self.mf_rho

    @property
    def mf_expected_wwT(self):
        """
        E[ww^T] = E_{A}[ E_{W|A}[ww^T | A] ]
                = rho * E[ww^T | A=1] + (1-rho) * 0
        :return:
        """
        mumuT = np.outer(self.mf_mu_w, self.mf_mu_w)
        return self.mf_rho * (self.mf_Sigma_w + mumuT)

    def meanfieldupdate(self):
        # Update mean and variance of Gaussian weight vector
        super(SpikeAndSlabGaussianVectorSynapse, self).meanfieldupdate()

        # Update the sparsity variational parameter
        logdet_Sigma_w = np.linalg.slogdet(self.Sigma_w)[1]
        logdet_mf_Sigma_w = np.linalg.slogdet(self.mf_Sigma_w)[1]
        mf_logit_rho = logit(self.rho) \
                       + self.D_in / 2.0 * (logdet_mf_Sigma_w - logdet_Sigma_w) \
                       + 0.5 * self.mf_mu_w.dot(np.linalg.solve(self.mf_Sigma_w, self.mf_mu_w)) \
                       - 0.5 * self.mu_w.dot(np.linalg.solve(self.Sigma_w, self.mu_w))

        self.mf_rho = logistic(mf_logit_rho)

    def get_vlb(self):
        """
        VLB for A and W
        :return:
        """
        vlb = 0

        # Precompute expectations
        # E_A            = self.network.mf_expected_p()
        # E_notA         = 1.0 - E_A
        # E_ln_rho       = self.network.mf_expected_log_p()
        # E_ln_notrho    = self.network.mf_expected_log_notp()
        E_A            = self.mf_rho
        E_notA         = 1.0 - E_A
        E_ln_rho       = np.log(self.rho)
        E_ln_notrho    = np.log(1.0 - self.rho)


        # E_mu           = self.network.expected_mu(self.n_pre, self.n_post)
        # E_mumuT        = self.network.expected_mumuT(self.n_pre, self.n_post)
        # E_Sigma_inv    = self.network.expected_Sigma_inv(self.n_pre, self.n_post)
        # E_logdet_Sigma = self.network.expected_logdet_Sigma(self.n_pre, self.n_post)
        E_W            = self.mf_expected_w
        E_WWT          = self.mf_expected_wwT
        E_mu           = self.mu_w
        E_mumuT        = self.mu_w.dot(self.mu_w.T)
        E_Sigma_inv    = np.linalg.inv(self.Sigma_w)
        E_logdet_Sigma = np.linalg.slogdet(self.Sigma_w)[1]


        # E[LN p(A | rho)]
        vlb += Bernoulli().negentropy(E_x=E_A, E_notx=E_notA,
                                      E_ln_p=E_ln_rho, E_ln_notp=E_ln_notrho).sum()

        # E[LN p(W | A=1, mu, Sigma)
        vlb += (E_A * Gaussian().negentropy(E_x=E_W, E_xxT=E_WWT,
                                            E_mu=E_mu, E_mumuT=E_mumuT,
                                            E_Sigma_inv=E_Sigma_inv, E_logdet_Sigma=E_logdet_Sigma))

        # E[LN q(W | A=1, mu, Sigma)
        vlb -= Bernoulli(self.mf_rho).negentropy()
        vlb -= E_A * Gaussian(self.mf_mu_w, self.mf_Sigma_w).negentropy()

        return vlb

    def resample_from_mf(self):
        super(SpikeAndSlabGaussianVectorSynapse, self).resample_from_mf()
        self.A = np.random.rand() < self.mf_rho

# TODO: Implement weighted, normalized synapses

# TODO: Implement synapses with learning rules

# TODO: Implement horseshoe, L1 synapses