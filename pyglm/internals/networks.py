"""
Network models expose a probability of connection and a scale of the weights
"""
import abc

import numpy as np
from scipy.special import gammaln, psi

from pyglm.abstractions import Component

from pyglm.deps.pybasicbayes.util.stats import sample_discrete_from_log
from pyglm.deps.pybasicbayes.util.stats import sample_niw

from pyglm.internals.distributions import Bernoulli

# Import graph models from graphistician
from pyglm.deps.graphistician.networks import GaussianWeightedEigenmodel

class Eigenmodel(Component):
    """
    Expose the GaussianWeightedEigenmodel through the Component interface
    """
    def __init__(self, population, D=2,
                 p=0.5, sigma_mu0=1.0, sigma_F=1.0,
                 lmbda=None, mu_lmbda=0, sigma_lmbda=1.0):
        self.population = population
        self.N = population.N
        self.B = population.B
        self.D = D

        # Instantiate the Gaussian weighted eigenmodel
        eigenmodel_args = {"p": p, "sigma_mu0": sigma_mu0, "sigma_F": sigma_F,
                           "lmbda": lmbda, "mu_lmbda": mu_lmbda, "sigma_lmbda": sigma_lmbda}
        self._model = GaussianWeightedEigenmodel(self.N, D, self.B,
                                                 eigenmodel_args=eigenmodel_args)

    @property
    def weight_model(self):
        return self.population.weight_model

    @property
    def P(self):
        """
        Get the NxN matrix of connection probabilities
        :return:
        """
        P = self._model.graph_model.P
        # if not self.allow_self_connections:
        #     np.fill_diagonal(P, 0.0)
        return P

    @property
    def Mu(self):
        """
        Get the NxNxB array of mean weights
        :return:
        """
        return np.tile(self._model.weight_model.mu[None, None, :],
                       [self.N, self.N, 1])

    @property
    def Sigma(self):
        """
        Get the NxNxBxB array of weight covariances
        :return:
        """
        return np.tile(self._model.weight_model.sigma[None, None, :, :],
                       [self.N, self.N, 1, 1])

    def log_prior(self):
        return self._model.log_prior()

    # Gibbs sampling
    def resample(self, augmented_data):
        self._model.resample((self.weight_model.A, self.weight_model.W))

    # Mean field
    def meanfieldupdate(self, augmented_data):
        E_A = self.weight_model.mf_expected_A()
        E_W = self.weight_model.mf_expected_w_given_A(A=1)
        E_WWT = self.weight_model.mf_expected_wwT_given_A(A=1)
        self._model.meanfieldupdate(E_A, E_W, E_WWT)

    def get_vlb(self, augmented_data):
        return self._model.get_vlb()

    def resample_from_mf(self, augmented_data):
        self._model.resample_from_mf()

    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        raise NotImplementedError()


# TODO: Move the SBM to graphistician

class _StochasticBlockModelBase(Component):
    """
    A stochastic block model is a clustered network model with
    K:          Number of nodes in the network
    C:          Number of blocks
    m[c]:       Probability that a node belongs block c
    p[c,c']:    Probability of connection from node in block c to node in block c'
    v[c,c']:    Scale of the gamma weight distribution from node in block c to node in block c'

    It is parameterized by:
    pi:         Parameter of Dirichlet prior over m
    tau0, tau1: Parameters of beta prior over p
    alpha:      Shape parameter of gamma prior over v
    beta:       Scale parameter of gamma prior over v
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, population,
                 C=1,
                 c=None, m=None, pi=1.0,
                 p=None, tau0=0.1, tau1=0.1,
                 mu=None, Sigma=None, mu0=0.0, kappa0=1.0, nu0=1.0, Sigma0=1.0,
                 allow_self_connections=True):
        """
        Initialize SBM with parameters defined above.
        """
        self.population = population
        self.N = self.population.N
        self.B = self.population.B

        assert isinstance(C, int) and C >= 1, "C must be a positive integer number of blocks"
        self.C = C

        if isinstance(pi, (int, float)):
            self.pi = pi * np.ones(C)
        else:
            assert isinstance(pi, np.ndarray) and pi.shape == (C,), "pi must be a sclar or a C-vector"
            self.pi = pi

        self.tau0    = tau0
        self.tau1    = tau1
        self.mu0     = mu0
        self.kappa0  = kappa0
        self.Sigma0  = Sigma0
        self.nu0     = nu0

        self.allow_self_connections = allow_self_connections

        if m is not None:
            assert isinstance(m, np.ndarray) and m.shape == (C,) \
                   and np.allclose(m.sum(), 1.0) and np.amin(m) >= 0.0, \
                "m must be a length C probability vector"
            self.m = m
        else:
            self.m = np.random.dirichlet(self.pi)


        if c is not None:
            assert isinstance(c, np.ndarray) and c.shape == (self.N,) and c.dtype == np.int \
                   and np.amin(c) >= 0 and np.amax(c) <= self.C-1, \
                "c must be a length K-vector of block assignments"
            self.c = c.copy()
        else:
            self.c = np.random.choice(self.C, p=self.m, size=(self.N))

        if p is not None:
            if np.isscalar(p):
                assert p >= 0 and p <= 1, "p must be a probability"
                self.p = p * np.ones((C,C))

            else:
                assert isinstance(p, np.ndarray) and p.shape == (C,C) \
                       and np.amin(p) >= 0 and np.amax(p) <= 1.0, \
                    "p must be a CxC matrix of probabilities"
                self.p = p
        else:
            self.p = np.random.beta(self.tau1, self.tau0, size=(self.C, self.C))

        if mu is not None and Sigma is not None:
            assert isinstance(mu, np.ndarray) and mu.shape == (C,C,self.B), \
                "mu must be a CxCxB array of mean weights"
            self.mu = mu

            assert isinstance(Sigma, np.ndarray) and Sigma.shape == (C,C,self.B,self.B), \
                "Sigma must be a CxCxBxB array of weight covariance matrices"
            self.sigma = Sigma

        else:
            # Sample from the normal inverse Wishart prior
            self.mu = np.zeros((C, C, self.B))
            self.sigma = np.zeros((C, C, self.B, self.B))
            for c1 in xrange(self.C):
                for c2 in xrange(self.C):
                    self.mu[c1,c2,:], self.sigma[c1,c2,:,:] = \
                        sample_niw(self.mu0, self.Sigma0, self.kappa0, self.nu0)

        # If m, p, and v are specified, then the model is fixed and the prior parameters
        # are ignored
        if None not in (c, p, mu, Sigma):
            self.fixed = True
        else:
            self.fixed = False

    @property
    def P(self):
        """
        Get the KxK matrix of probabilities
        :return:
        """
        P = self.p[np.ix_(self.c, self.c)]
        if not self.allow_self_connections:
            np.fill_diagonal(P, 0.0)
        return P

    @property
    def Mu(self):
        """
        Get the NxNxB array of mean weights
        :return:
        """
        return self.mu[np.ix_(self.c, self.c, np.arange(self.B))]

    @property
    def Sigma(self):
        """
        Get the NxNxB array of mean weights
        :return:
        """
        return self.sigma[np.ix_(self.c, self.c, np.arange(self.B), np.arange(self.B))]

    # def log_likelihood(self, x):
    #     """
    #     Compute the log likelihood of a set of SBM parameters
    #
    #     :param x:    (m,p,v) tuple
    #     :return:
    #     """
    #     m,p,v,c = x
    #
    #     lp = 0
    #     lp += Dirichlet(self.pi).log_probability(m)
    #     lp += Beta(self.tau1 * np.ones((self.C, self.C)),
    #                self.tau0 * np.ones((self.C, self.C))).log_probability(p).sum()
    #     lp += Gamma(self.mu0, self.Sigma0).log_probability(v).sum()
    #     lp += (np.log(m)[c]).sum()
    #     return lp
    #
    # def log_probability(self):
    #     return self.log_likelihood((self.m, self._p, self.v, self.c))

class _GibbsSBM(_StochasticBlockModelBase):
    """
    Implement Gibbs sampling for SBM
    """
    def __init__(self, population,
                 C=1,
                 c=None, pi=1.0, m=None,
                 p=None, tau0=0.1, tau1=0.1,
                 mu=None, Sigma=None, mu0=0.0, kappa0=1.0, nu0=1.0, Sigma0=1.0,
                 allow_self_connections=True):

        super(_GibbsSBM, self).__init__(population,
                                        C=C,
                                        c=c, pi=pi, m=m,
                                        p=p, tau0=tau0, tau1=tau1,
                                        mu=mu, Sigma=Sigma, mu0=mu0, kappa0=kappa0, Sigma0=Sigma0, nu0=nu0,
                                        allow_self_connections=allow_self_connections)

        # Initialize parameter estimates
        # print "Uncomment GibbsSBM init"
        # if not self.fixed:
        #     self.c = np.random.choice(self.C, size=(self.N))
        #     self.m = 1.0/C * np.ones(self.C)
        #     # self.p = self.tau1 / (self.tau0 + self.tau1) * np.ones((self.C, self.C))
        #     self._p = np.random.beta(self.tau1, self.tau0, size=(self.C, self.C))
        #     # self.v = self.alpha / self.beta * np.ones((self.C, self.C))
        #     self.v = np.random.gamma(self.mu0, 1.0/self.Sigma0, size=(self.C, self.C))

    def resample_p(self, A):
        """
        Resample p given observations of the weights
        """
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                Ac1c2 = A[np.ix_(self.c==c1, self.c==c2)]

                if not self.allow_self_connections:
                    # TODO: Account for self connections
                    pass

                tau1 = self.tau1 + Ac1c2.sum()
                tau0 = self.tau0 + (1-Ac1c2).sum()
                self.p[c1,c2] = np.random.beta(tau1, tau0)

    def resample_mu_Sigma(self, A, W):
        """
        Resample v given observations of the weights
        """
        # import pdb; pdb.set_trace()
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                Ac1c2 = A[np.ix_(self.c==c1, self.c==c2)]
                Wc1c2 = W[np.ix_(self.c==c1, self.c==c2)]
                # alpha = self.mu0 + Ac1c2.sum() * self.kappa
                # beta  = self.Sigma0 + Wc1c2[Ac1c2 > 0].sum()
                # self.v[c1,c2] = np.random.gamma(alpha, 1.0/beta)

    def resample_c(self, A, W):
        """
        Resample block assignments given the weighted adjacency matrix
        and the impulse response fits (if used)
        """
        if self.C == 1:
            return

        # Sample each assignment in order
        for k in xrange(self.N):
            # Compute unnormalized log probs of each connection
            lp = np.zeros(self.C)

            # Prior from m
            lp += np.log(self.m)

            # Likelihood from network
            for ck in xrange(self.C):
                c_temp = self.c.copy().astype(np.int)
                c_temp[k] = ck

                # p(A[k,k'] | c)
                lp[ck] += Bernoulli(self.p[ck, c_temp])\
                                .log_probability(A[k,:]).sum()

                # p(A[k',k] | c)
                lp[ck] += Bernoulli(self.p[c_temp, ck])\
                                .log_probability(A[:,k]).sum()

                # p(W[k,k'] | c)
                # lp[ck] += (A[k,:] * Gamma(self.kappa, self.v[ck, c_temp])\
                #                 .log_probability(W[k,:])).sum()
                #
                # # p(W[k,k'] | c)
                # lp[ck] += (A[:,k] * Gamma(self.kappa, self.v[c_temp, ck])\
                #                 .log_probability(W[:,k])).sum()

                # TODO: Subtract of self connection since we double counted

                # TODO: Get probability of impulse responses g

            # Resample from lp
            self.c[k] = sample_discrete_from_log(lp)

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)

    def resample(self, augmented_data):
        return

        # if self.fixed:
        #     return
        #
        # A = self.population.weight_model.A
        # W = self.population.weight_model.W
        #
        # self.resample_p(A)
        # self.resample_mu_Sigma(A, W)
        # self.resample_c(A, W)
        # self.resample_m()


class _MeanFieldSBM(_StochasticBlockModelBase):
    """
    Add mean field updates
    """
    def __init__(self, population,
                 C=1,
                 c=None, pi=1.0, m=None,
                 p=None, tau0=0.1, tau1=0.1,
                 mu=None, Sigma=None, mu0=0.0, kappa0=1.0, nu0=1.0, Sigma0=1.0,
                 allow_self_connections=True):

        super(_MeanFieldSBM, self).__init__(population,
                                            C=C,
                                            c=c, pi=pi, m=m,
                                            p=p, tau0=tau0, tau1=tau1,
                                            mu=mu, Sigma=Sigma, mu0=mu0, kappa0=kappa0, Sigma0=Sigma0, nu0=nu0,
                                            allow_self_connections=allow_self_connections)

        # Initialize mean field parameters
        self.mf_pi     = np.ones(self.C)

        # To break symmetry, start with a sample of mf_m
        self.mf_m      = np.random.dirichlet(10 * np.ones(self.C),
                                            size=(self.N,))
        self.mf_tau0   = self.tau0  * np.ones((self.C, self.C))
        self.mf_tau1   = self.tau1  * np.ones((self.C, self.C))
        self.mf_Mu     = self.Mu.copy()
        self.mf_Sigma  = self.Sigma.copy()

    def meanfieldupdate(self, augmented_data):
        raise NotImplementedError()

    def mf_update_c(self, E_A, E_notA, E_W_given_A, E_ln_W_given_A, stepsize=1.0):
        """
        Update the block assignment probabilitlies one at a time.
        This one involves a number of not-so-friendly expectations.
        :return:
        """
        # Sample each assignment in order
        for n1 in xrange(self.N):
            notk = np.concatenate((np.arange(n1), np.arange(n1+1,self.N)))

            # Compute unnormalized log probs of each connection
            lp = np.zeros(self.C)

            # Prior from m
            lp += self.expected_log_m()

            # Likelihood from network
            for ck in xrange(self.C):

                # Compute expectations with respect to other block assignments, c_{\neg k}
                # Initialize vectors for expected parameters
                E_ln_p_ck_to_cnotk    = np.zeros(self.N-1)
                E_ln_notp_ck_to_cnotk = np.zeros(self.N-1)
                E_ln_p_cnotk_to_ck    = np.zeros(self.N-1)
                E_ln_notp_cnotk_to_ck = np.zeros(self.N-1)
                E_v_ck_to_cnotk       = np.zeros(self.N-1)
                E_ln_v_ck_to_cnotk    = np.zeros(self.N-1)
                E_v_cnotk_to_ck       = np.zeros(self.N-1)
                E_ln_v_cnotk_to_ck    = np.zeros(self.N-1)

                for cnotk in xrange(self.C):
                    # Get the (K-1)-vector of other class assignment probabilities
                    p_cnotk = self.mf_m[notk,cnotk]

                    # Expected log probability of a connection from ck to cnotk
                    E_ln_p_ck_to_cnotk    += p_cnotk * (psi(self.mf_tau1[ck, cnotk])
                                                        - psi(self.mf_tau0[ck, cnotk] + self.mf_tau1[ck, cnotk]))
                    E_ln_notp_ck_to_cnotk += p_cnotk * (psi(self.mf_tau0[ck, cnotk])
                                                        - psi(self.mf_tau0[ck, cnotk] + self.mf_tau1[ck, cnotk]))

                    # Expected log probability of a connection from cnotk to ck
                    E_ln_p_cnotk_to_ck    += p_cnotk * (psi(self.mf_tau1[cnotk, ck])
                                                        - psi(self.mf_tau0[cnotk, ck] + self.mf_tau1[cnotk, ck]))
                    E_ln_notp_cnotk_to_ck += p_cnotk * (psi(self.mf_tau0[cnotk, ck])
                                                        - psi(self.mf_tau0[cnotk, ck] + self.mf_tau1[cnotk, ck]))

                    # Expected log scale of connections from ck to cnotk
                    E_v_ck_to_cnotk       += p_cnotk * (self.mf_alpha[ck, cnotk] / self.mf_beta[ck, cnotk])
                    E_ln_v_ck_to_cnotk    += p_cnotk * (psi(self.mf_alpha[ck, cnotk])
                                                        - np.log(self.mf_beta[ck, cnotk]))

                    # Expected log scale of connections from cnotk to ck
                    E_v_cnotk_to_ck       += p_cnotk * (self.mf_alpha[cnotk, ck] / self.mf_beta[cnotk, ck])
                    E_ln_v_cnotk_to_ck    += p_cnotk * (psi(self.mf_alpha[cnotk, ck])
                                                        - np.log(self.mf_beta[cnotk, ck]))

                # Compute E[ln p(A | c, p)]
                lp[ck] += Bernoulli().negentropy(E_x=E_A[n1, notk],
                                                 E_notx=E_notA[n1, notk],
                                                 E_ln_p=E_ln_p_ck_to_cnotk,
                                                 E_ln_notp=E_ln_notp_ck_to_cnotk).sum()

                lp[ck] += Bernoulli().negentropy(E_x=E_A[notk, n1],
                                                 E_notx=E_notA[notk, n1],
                                                 E_ln_p=E_ln_p_cnotk_to_ck,
                                                 E_ln_notp=E_ln_notp_cnotk_to_ck).sum()

                # Compute E[ln p(W | A=1, c, v)]
                lp[ck] += (E_A[n1, notk] *
                           Gamma(self.kappa).negentropy(E_ln_lambda=E_ln_W_given_A[n1, notk],
                                                        E_lambda=E_W_given_A[n1,notk],
                                                        E_beta=E_v_ck_to_cnotk,
                                                        E_ln_beta=E_ln_v_ck_to_cnotk)).sum()

                lp[ck] += (E_A[n1, notk] *
                           Gamma(self.kappa).negentropy(E_ln_lambda=E_ln_W_given_A[notk, n1],
                                                        E_lambda=E_W_given_A[notk,n1],
                                                        E_beta=E_v_cnotk_to_ck,
                                                        E_ln_beta=E_ln_v_cnotk_to_ck)).sum()

                # Compute expected log prob of self connection
                if self.allow_self_connections:
                    E_ln_p_ck_to_ck    = psi(self.mf_tau1[ck, ck]) - psi(self.mf_tau0[ck, ck] + self.mf_tau1[ck, ck])
                    E_ln_notp_ck_to_ck = psi(self.mf_tau0[ck, ck]) - psi(self.mf_tau0[ck, ck] + self.mf_tau1[ck, ck])
                    lp[ck] += Bernoulli().negentropy(E_x=E_A[n1, n1],
                                                     E_notx=E_notA[n1, n1],
                                                     E_ln_p=E_ln_p_ck_to_ck,
                                                     E_ln_notp=E_ln_notp_ck_to_ck
                                                    )
                    E_v_ck_to_ck    = self.mf_alpha[ck, ck] / self.mf_beta[ck, ck]
                    E_ln_v_ck_to_ck = psi(self.mf_alpha[ck, ck]) - np.log(self.mf_beta[ck, ck])
                    lp[ck] += (E_A[n1, n1] *
                               Gamma(self.kappa).negentropy(E_ln_lambda=E_ln_W_given_A[n1, n1],
                                                            E_lambda=E_W_given_A[n1,n1],
                                                            E_beta=E_v_ck_to_ck,
                                                            E_ln_beta=E_ln_v_ck_to_ck))


                # TODO: Get probability of impulse responses g


            # Normalize the log probabilities to update mf_m
            Z = logsumexp(lp)
            mk_hat = np.exp(lp - Z)

            self.mf_m[n1,:] = (1.0 - stepsize) * self.mf_m[n1,:] + stepsize * mk_hat

    def mf_expected_p(self):
        """
        Compute the expected probability of a connection, averaging over c
        :return:
        """
        if self.fixed:
            return self.P

        E_p = np.zeros((self.N, self.N))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_p += pc1c2 * self.mf_tau1[c1,c2] / (self.mf_tau0[c1,c2] + self.mf_tau1[c1,c2])

        if not self.allow_self_connections:
            np.fill_diagonal(E_p, 0.0)

        return E_p

    def mf_expected_notp(self):
        """
        Compute the expected probability of NO connection, averaging over c
        :return:
        """
        return 1.0 - self.expected_p()

    def mf_expected_log_p(self):
        """
        Compute the expected log probability of a connection, averaging over c
        :return:
        """
        if self.fixed:
            E_ln_p = np.log(self.P)
        else:
            E_ln_p = np.zeros((self.N, self.N))
            for c1 in xrange(self.C):
                for c2 in xrange(self.C):
                    # Get the KxK matrix of joint class assignment probabilities
                    pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                    # Get the probability of a connection for this pair of classes
                    E_ln_p += pc1c2 * (psi(self.mf_tau1[c1,c2])
                                       - psi(self.mf_tau0[c1,c2] + self.mf_tau1[c1,c2]))

        if not self.allow_self_connections:
            np.fill_diagonal(E_ln_p, -np.inf)

        return E_ln_p

    def mf_expected_log_notp(self):
        """
        Compute the expected log probability of NO connection, averaging over c
        :return:
        """
        if self.fixed:
            E_ln_notp = np.log(1.0 - self.P)
        else:
            E_ln_notp = np.zeros((self.N, self.N))
            for c1 in xrange(self.C):
                for c2 in xrange(self.C):
                    # Get the KxK matrix of joint class assignment probabilities
                    pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                    # Get the probability of a connection for this pair of classes
                    E_ln_notp += pc1c2 * (psi(self.mf_tau0[c1,c2])
                                          - psi(self.mf_tau0[c1,c2] + self.mf_tau1[c1,c2]))

        if not self.allow_self_connections:
            np.fill_diagonal(E_ln_notp, 0.0)

        return E_ln_notp

    def mf_expected_mu(self):
        # TODO: Use variational parameters
        E_mu = self.Mu
        return E_mu

    def mf_expected_mumuT(self):
        # TODO: Use variational parameters
        E_mumuT = np.einsum("ijk,ijl->ijkl", self.Mu, self.Mu)

        return E_mumuT

    def mf_expected_Sigma_inv(self):
        # TODO: Use variational parameters
        E_Sigma_inv = self.Sigma.copy()
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                E_Sigma_inv[n_pre, n_post, :, :] = \
                    np.linalg.inv(E_Sigma_inv[n_pre, n_post, :, :])

        return E_Sigma_inv

    def mf_expected_logdet_Sigma(self):
        # TODO: Use variational parameters
        Sigma = self.Sigma
        E_logdet_Sigma = np.zeros((self.N, self.N))
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                E_logdet_Sigma[n_pre, n_post] = \
                    np.linalg.slogdet(Sigma[n_pre, n_post, :, :])[1]
        return E_logdet_Sigma

    def get_vlb(self,
                vlb_c=True,
                vlb_p=True,
                vlb_v=True,
                vlb_m=True):
        # import pdb; pdb.set_trace()
        vlb = 0

        # Get the VLB of the expected class assignments
        if vlb_c:
            E_ln_m = self.expected_log_m()
            for k in xrange(self.K):
                # Add the cross entropy of p(c | m)
                vlb += Discrete().negentropy(E_x=self.mf_m[k,:], E_ln_p=E_ln_m)

                # Subtract the negative entropy of q(c)
                vlb -= Discrete(self.mf_m[k,:]).negentropy()

        # Get the VLB of the connection probability matrix
        # Add the cross entropy of p(p | tau1, tau0)
        if vlb_p:
            vlb += Beta(self.tau1, self.tau0).\
                negentropy(E_ln_p=(psi(self.mf_tau1) - psi(self.mf_tau0 + self.mf_tau1)),
                           E_ln_notp=(psi(self.mf_tau0) - psi(self.mf_tau0 + self.mf_tau1))).sum()

            # Subtract the negative entropy of q(p)
            vlb -= Beta(self.mf_tau1, self.mf_tau0).negentropy().sum()

        # Get the VLB of the weight scale matrix, v
        # Add the cross entropy of p(v | alpha, beta)
        if vlb_v:
            vlb += Gamma(self.alpha, self.beta).\
                negentropy(E_lambda=self.mf_alpha/self.mf_beta,
                           E_ln_lambda=psi(self.mf_alpha) - np.log(self.mf_beta)).sum()

            # Subtract the negative entropy of q(v)
            vlb -= Gamma(self.mf_alpha, self.mf_beta).negentropy().sum()

        # Get the VLB of the block probability vector, m
        # Add the cross entropy of p(m | pi)
        if vlb_m:
            vlb += Dirichlet(self.pi).negentropy(E_ln_g=self.expected_log_m())

            # Subtract the negative entropy of q(m)
            vlb -= Dirichlet(self.mf_pi).negentropy()

        return vlb

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        self.m = np.random.dirichlet(self.mf_pi)
        self.p = np.random.beta(self.mf_tau1, self.mf_tau0)
        self.v = np.random.gamma(self.mf_alpha, 1.0/self.mf_beta)

        self.c = np.zeros(self.K, dtype=np.int)
        for k in xrange(self.K):
            self.c[k] = int(np.random.choice(self.C, p=self.mf_m[k,:]))

    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        raise NotImplementedError()

class StochasticBlockModel(_GibbsSBM, _MeanFieldSBM):
    pass