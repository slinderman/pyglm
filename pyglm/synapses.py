import numpy as np
from numpy import newaxis as na

from pyglm.deps.pybasicbayes.util.general import blockarray
from pyglm.deps.pybasicbayes.abstractions import Collapsed
from pyglm.deps.pybasicbayes.distributions import GibbsSampling, GaussianFixed


class GaussianVectorSynapse(GibbsSampling, Collapsed):
    def __init__(self,
                 neuron_model,
                 n_pre,
                 sigma,
                 affine=False,
                 w=None):

        self.n_pre = n_pre
        self.affine = affine
        self.sigma = sigma
        self.w = w

        # Create cache for X \dot w
        # self.cache = SimpleCache()
        self.cache = None

        self.neuron_model = neuron_model

        # if Sigma_w is not None:
        #     self.Sigma_w_inv = np.linalg.inv(Sigma_w)
        # else:
        #     self.Sigma_w_inv = None

        if w is None:
            assert self.mu_w.ndim == 1
            self.resample() # initialize from prior

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
        # NOTE: D_in includes the extra affine coordinate
        mat = self.w if self.w is not None else self.mu_w
        return mat.shape[0]

    @property
    def D_out(self):
        # For now, assume we are doing scalar regression
        return 1

    def set_weights(self, value):
        self.w = value
        self.cache = None

    ### getting statistics

    def _get_statistics(self,data):
        if isinstance(data,list):
            return sum((self._get_statistics(d) for d in data),self._empty_statistics())
        else:
            data = data[~np.isnan(data).any(1)]
            n, D = data.shape[0], self.D_out

            statmat = data.T.dot(data)
            xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]

            if self.affine:
                xy = data.sum(0)
                x, y = xy[:-D], xy[-D:]
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])

    def _empty_statistics(self):
        return np.array([np.zeros((self.D_out, self.D_out)),
                         np.zeros((1,self.D_in)),
                         np.zeros((self.D_in, self.D_in)),
                         0])

    ### distribution

    def log_likelihood(self,xy):
        w, sigma, D = self.w, self.sigma, self.D_out
        x, y = xy[:,:-D], xy[:,-D:]

        if self.affine:
            w, b = w[:-1], w[-1]
            mu_y = x.dot(w.T) + b
        else:
            mu_y = x.dot(w.T)

        ll = (-0.5/sigma * (y-mu_y)**2).sum()
        ll -= -0.5 * np.log(2*np.pi * sigma**2)

        return ll

    def rvs(self,x=None,size=1,return_xy=True):
        w, sigma = self.w, self.sigma

        if self.affine:
            w, b = w[:-1], w[-1]

        x = np.random.normal(size=(size,w.shape[1])) if x is None else x
        y = x.dot(w.T) + np.sqrt(sigma) * np.random.normal(size=(x.shape[0],))

        if self.affine:
            y += b.T

        return np.hstack((x,y[:,None])) if return_xy else y

    ### Gibbs sampling
    def cond_w(self, data=[], stats=None):
        ss = self._get_statistics(data) if stats is None else stats
        yxT = ss[1]
        xxT = ss[2]

        # Posterior mean of a Gaussian
        Sigma_w_inv = np.linalg.inv(self.Sigma_w)
        Sigma_w_post = np.linalg.inv(xxT / self.neuron_model.sigma + Sigma_w_inv)
        mu_w_post = ((yxT/self.neuron_model.sigma + self.mu_w.dot(Sigma_w_inv)).dot(Sigma_w_post)).reshape((self.D_in,))
        return GaussianFixed(mu_w_post, Sigma_w_post)

    def resample(self,data=[],stats=None):
        dist = self.cond_w(data=data, stats=stats)
        w = dist.rvs()[0,:]
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
        if self.affine:
            w, b = w[:-1], w[-1]

        y = X.dot(w.T)
        if self.affine:
            y += b.T

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
            d = np.asscalar(self.sigma) * np.ones((N,))
            Ainv = 1.0/d

            # Sig_marg_inv = invert_low_rank(Ainv, X, self.Sigma_A, X.T, diag=True)
            yy = y-mu_marg
            tmp = -1./2. * quad_form_diag_plus_lr2(yy, d, X, self.Sigma_w, X.T)
            out = tmp \
                  - N/2*np.log(2*np.pi) \
                  - 0.5 * logdet_low_rank2(Ainv, X, self.Sigma_w, X.T, diag=True)

            # Debug the marginal likelihood calculation
            #
            # Compute the marginal distribution parameters
            # mu_marg = X.dot(self.mu_w.T).reshape((N,))
            # # Covariances add
            # Sig_marg = np.asscalar(self.sigma) * np.eye(N) + X.dot(self.Sigma_w.dot(X.T))
            # # Compute the marginal log likelihood
            # out2 = GaussianFixed(mu_marg, Sig_marg).log_likelihood(y)
            # assert np.allclose(out, out2)
        else:
            raise Exception("Data must be list of numpy arrays or numpy array")

        return out


# TODO: Implement weighted, normalized synapses

# TODO: Implement synapses with learning rules

# TODO: Implement horseshoe, L1 synapses