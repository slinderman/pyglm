import numpy as np
import scipy.special as special
from pyglm.deps.pybasicbayes.abstractions import GibbsSampling
from pyglm.deps.pybasicbayes.util.stats import getdatasize

class InverseGamma(GibbsSampling):
    """
    Base class for an inverse gamma prior on the variance of a scalar Gaussian
    """
    def __init__(self, sigma=None, alpha_0=1.0, beta_0=1.0):

        self.sigma=sigma
        self.alpha_0, self.beta_0 = alpha_0, beta_0

        if sigma is None:
            assert None not in [alpha_0, beta_0]
            self.resample()

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
