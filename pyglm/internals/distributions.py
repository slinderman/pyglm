import numpy as np
import scipy.special as special
from pyglm.deps.pybasicbayes.abstractions import GibbsSampling, MeanField
from pyglm.deps.pybasicbayes.util.stats import getdatasize

class InverseGamma(GibbsSampling, MeanField):
    """
    Base class for an inverse gamma prior on the variance of a scalar Gaussian
    """
    def __init__(self, sigma=None, alpha_0=1.0, beta_0=1.0):

        self.sigma=sigma
        self.alpha_0, self.beta_0 = alpha_0, beta_0

        if sigma is None:
            assert None not in [alpha_0, beta_0]
            self.resample()

        # Initialize mean field parameters
        self.mf_alpha_0 = alpha_0
        self.mf_beta_0  = beta_0

    def rvs(self,size=None):
        'random variates (samples)'
        if size is None:
            size = (1,)

        g = np.random.gamma(self.alpha_0, 1.0/self.beta_0, *size)
        return 1./g

    def log_likelihood(self,x):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        alpha_0, beta_0 = self.alpha_0, self.beta_0
        return alpha_0 * np.log(beta_0) - special.gammaln(alpha_0) + \
               -(alpha_0-1) * np.log(x) -beta_0 / x

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data, np.ndarray):
                xss = (data**2).sum()
            else:
                xss = np.sum([(d**2).sum() for d in data])
        else:
            xss = 0
        return n, xss

    def _posterior_hypparams(self,n,xss):
        alpha_0, beta_0 = self.alpha_0, self.beta_0
        alpha_n = alpha_0 + n/2.0
        beta_n = beta_0 + xss/2.0

        return alpha_n, beta_n

    def resample(self,data=[]):
        """
        Resample the variance, sigma^2, given observations of x-mu, i.e. the residuals,
        where x~N(mu, sigma^2).

        :param data: a vector or residuals, x-mu
        """
        alpha_n, beta_n = self._posterior_hypparams(*self._get_statistics(data))
        self.sigma = 1.0/np.random.gamma(alpha_n, 1.0/beta_n)

    ### Mean Field
    def expected_log_likelihood(self,x):
        raise NotImplementedError()

    def expected_eta_inv(self):
        return self.mf_alpha_0 / self.mf_beta_0

    def expected_log_eta(self):
        return np.log(self.mf_beta_0) - special.psi(self.mf_alpha_0)

    def meanfieldupdate(self,data,weights):
        raise NotImplementedError()

    def negentropy(self, E_ln_eta=None, E_eta_inv=None, alpha=None, E_beta=None, E_ln_beta=None):
        """
        Compute the entropy of the inverse gamma distribution.
        :param E_ln_eta:    If given, use this in place of expectation wrt alpha and beta
        :param E_eta_inv:       If given, use this in place of expectation wrt alpha and beta
        :param E_ln_beta:   If given, use this in place of expectation wrt alpha and beta
        :param E_beta:      If given, use this in place of expectation wrt alpha and beta
        :return: E[ ln p(\lambda | \alpha, \beta)]
        """
        if E_ln_eta is None:
            E_ln_eta = self.expected_log_eta()

        if E_eta_inv is None:
            E_eta_inv = self.expected_eta_inv()

        if E_ln_beta is None:
            E_ln_beta = np.log(self.beta_0)

        if E_beta is None:
            E_beta = self.beta_0

        if alpha is None:
            alpha = self.alpha_0

        # Compute the expected log prob
        H =  alpha * E_ln_beta
        H += -special.gammaln(alpha)
        H += (-alpha - 1.0) * E_ln_eta
        H += -E_beta * E_eta_inv

        return H

    def get_vlb(self):
        """
        Compute the variational lower bound for eta
        E[LN p(eta | alpha, beta)] - E[LN q(eta | mf_alpha, mf_beta)]
        where both p and q are inverse gamma distributions
        :return:
        """
        # TODO: Move the InverseGamma to an activation noise class
        vlb = 0

        # E[LN p(eta | alpha, beta)]
        E_eta_inv = self.expected_eta_inv()
        E_ln_eta  = self.expected_log_eta()
        vlb += self.negentropy(E_ln_eta=E_ln_eta, E_eta_inv=E_eta_inv)

        # - E[LN q(eta | mf_alpha, mf_beta)]
        vlb -= self.negentropy(alpha=self.mf_alpha_0, E_beta=self.mf_beta_0, E_ln_beta=np.log(self.mf_beta_0))

        return vlb


class Bernoulli:
    #TODO: Subclass Discrete distribution
    def __init__(self, p=0.5):
        assert np.all(p >= 0) and np.all(p <= 1.0)
        self.p = p

    def log_probability(self, x):
        """
        Log probability of x given p
        :param x:
        :return:
        """
        lp = x * np.log(self.p) + (1-x) * np.log(1.0-self.p)
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.p

    def expected_notx(self):
        return 1 - self.p

    def negentropy(self, E_x=None, E_notx=None, E_ln_p=None, E_ln_notp=None):
        """
        Compute the entropy of the Bernoulli distribution.
        :param E_x:         If given, use this in place of expectation wrt p
        :param E_notx:      If given, use this in place of expectation wrt p
        :param E_ln_p:      If given, use this in place of expectation wrt p
        :param E_ln_notp:   If given, use this in place of expectation wrt p
        :return: E[ ln p(x | p)]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_notx is None:
            E_notx = self.expected_notx()

        if E_ln_p is None:
            E_ln_p = np.log(self.p)

        if E_ln_notp is None:
            E_ln_notp = np.log(1.0 - self.p)

        H = E_x * E_ln_p + E_notx * E_ln_notp
        H = np.nan_to_num(H)
        return H


class ScalarGaussian:

    def __init__(self, mu=0.0, sigmasq=1.0):
        assert np.all(sigmasq) >= 0
        self.mu = mu
        self.sigmasq = sigmasq

    def log_probability(self, x):
        """
        Log probability of x given mu, sigmasq
        :param x:
        :return:
        """
        lp = -0.5*np.log(2*np.pi*self.sigmasq) -1.0/(2*self.sigmasq) * (x-self.mu)**2
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.mu

    def expected_xsq(self):
        return self.sigmasq + self.mu**2


    def negentropy(self, E_x=None, E_xsq=None, E_mu=None, E_musq=None, E_sigmasq_inv=None, E_ln_sigmasq=None):
        """
        Compute the negative entropy of the Gaussian distribution
        :return: E[ ln p(x | mu, sigmasq)] = E[-0.5*log(2*pi*sigmasq) - 0.5/sigmasq * (x-mu)**2]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_xsq is None:
            E_xsq = self.expected_xsq()

        if E_mu is None:
            E_mu = self.mu

        if E_musq is None:
            E_musq = self.mu**2

        if E_sigmasq_inv is None:
            E_sigmasq_inv = 1.0/self.sigmasq

        if E_ln_sigmasq is None:
            E_ln_sigmasq = np.log(self.sigmasq)

        H  = -0.5 * np.log(2*np.pi)
        H += -0.5 * E_ln_sigmasq
        H += -0.5 * E_sigmasq_inv * E_xsq
        H += E_sigmasq_inv * E_x * E_mu
        H += -0.5 * E_sigmasq_inv * E_musq
        return H


class Gaussian:

    def __init__(self, mu=np.zeros(1), Sigma=np.eye(1)):
        assert mu.ndim == 1, "Mu must be a 1D vector"
        self.mu = mu
        self.D  = mu.shape[0]

        assert Sigma.shape == (self.D, self.D), "Sigma must be a DxD covariance matrix"
        self.Sigma = Sigma

    def log_probability(self, x):
        """
        Log probability of x given mu, sigmasq
        :param x:
        :return:
        """
        logdet_Sigma = np.linalg.slogdet(self.Sigma)[1]
        z = x-self.mu
        lp = -0.5*np.log(2*np.pi) -0.5*logdet_Sigma -0.5 * z.T.dot(self.Sigma).dot(z)
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.mu

    def expected_xxT(self):
        return self.Sigma + np.outer(self.mu, self.mu)


    def negentropy(self, E_x=None, E_xxT=None, E_mu=None, E_mumuT=None, E_Sigma_inv=None, E_logdet_Sigma=None):
        """
        Compute the negative entropy of the Gaussian distribution
        :return: E[ ln p(x | mu, sigmasq)] = E[-0.5*log(2*pi) -0.5*E[log |Sigma|] - 0.5 * (x-mu)^T Sigma^{-1} (x-mu)]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_xxT is None:
            E_xxT = self.expected_xxT()

        if E_mu is None:
            E_mu = self.mu

        if E_mumuT is None:
            E_mumuT = np.outer(self.mu, self.mu)

        if E_Sigma_inv is None:
            E_Sigma_inv = np.linalg.inv(self.Sigma)

        if E_logdet_Sigma is None:
            E_logdet_Sigma = np.linalg.slogdet(self.Sigma)[1]

        H  = -0.5 * np.log(2*np.pi)
        H += -0.5 * E_logdet_Sigma
        # TODO: Replace trace with something more efficient
        H += -0.5 * np.trace(E_Sigma_inv.dot(E_xxT))
        H += E_x.T.dot(E_Sigma_inv).dot(E_mu)
        H += -0.5 * np.trace(E_Sigma_inv.dot(E_mumuT))
        return H