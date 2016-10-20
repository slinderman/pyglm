import numpy as np
from pybasicbayes.abstractions import Model

import pyglm.regression
from pyglm.utils.basis import convolve_with_basis

class NonlinearAutoregressiveModel(Model):
    """
    The "generalized linear model" in neuroscience is really
    a vector autoregressive model.

    # TODO: Finish description
    """

    def __init__(self, N, regressions, basis=None, B=10):
        """
        :param N:             Observation dimension
        :param regressions:   Regression objects, one per observation dim.
        :param basis:         Basis onto which the preceding activity is projected
                              In the "identity" case, this is just a lag matrix
        :param B:             Basis dimensionality.
                              In the "identity" case, this is the number of lags.
        """
        self.N = N

        assert len(regressions) == N
        self.regressions = regressions

        # Initialize the basis
        if basis is None:
            basis = np.eye(B)
        else:
            assert basis.ndim == 2
        self.basis = basis
        self.B = self.basis.shape[1]

        # Initialize the data list to empty
        self.data_list = []

    # Expose the autoregressive weights and adjacency matrix
    @property
    def weights(self):
        return np.array([r.W[0] for r in self.regressions])

    @property
    def adjacency(self):
        return np.array([r.a for r in self.regressions])

    @property
    def biases(self):
        return np.array([r.b for r in self.regressions]).ravel()

    def add_data(self, data, X=None):
        N, B = self.N, self.B
        assert isinstance(data, np.ndarray) \
               and data.ndim == 2 \
               and data.shape[1] == self.N
        T = data.shape[0]

        # Convolve the data with the basis to get regressors
        if X is None:
            X = convolve_with_basis(data, self.basis)
        else:
            assert X.shape == (T, N, B)

        # Add the covariates and observations
        self.data_list.append((X, data))

    def log_likelihood(self, datas=None):
        if datas is None:
            datas = self.data_list

        ll = 0
        for data in datas:
            if isinstance(data, tuple):
                X, Y = data
            else:
                X, Y = convolve_with_basis(data, self.basis), data

            for n, reg in enumerate(self.regressions):
                ll += reg.log_likelihood((X, Y[:,n])).sum()

        return ll

    def generate(self, keep=True, T=100, verbose=False, print_intvl=10):
        """
        Generate data from the model.

        :param keep:    Add the data to the model's datalist
        :param T:       Number of time bins to simulate
        :return:
        """
        if T == 0:
            return np.zeros((0,self.N))
        assert isinstance(T, int), "Size must be an integer number of time bins"

        N, basis = self.N, self.basis
        L, B = basis.shape

        # NOTE: to be consistent with 'convolve_with_basis', we have
        # to flip the basis left to right here
        basis = np.flipud(basis)
        assert not np.allclose(basis, self.basis)

        # Precompute the weights and biases
        W = self.weights.reshape((N, N*B))  # N x NB (post x (pre x B))
        b = self.biases                     # N (post)

        # Initialize output matrix of spike counts
        Y = np.zeros((T+L, N))
        X = np.zeros((T+L, N, B))
        Psi = np.zeros((T+L, N))

        # Iterate forward in time
        for t in range(L,T+L):
            if verbose:
                if t % print_intvl == 0:
                    print("Generate t={}".format(t))
            # 1. Project previous activity window onto the basis
            #    previous activity is L x N, basis is L x B,
            X[t] = Y[t-L:t].T.dot(basis)

            # 2. Compute the activation, W.dot(X[t]) + b
            Psi[t] = W.dot(X[t].reshape((N*B,))) + b

            # 3. Weight the previous activity with the regression weights
            Y[t] = self.regressions[0].rvs(psi=Psi[t])

            # for n, reg in enumerate(self.regressions):
            #     Y[t,n] = reg.rvs(X=X[t:t+1])

        if keep:
            self.add_data(Y[L:], X=X[L:])
            # self.add_data(Y[L:])

        return X[L:], Y[L:]

    @property
    def means(self):
        """
        Compute the mean observation for each dataset
        """
        mus = []
        for (X,Y) in self.data_list:
            mus.append(np.column_stack(
                [r.mean(X) for r in self.regressions]))

        return mus

    ### Gibbs sampling
    def resample_model(self):
        self.resample_regressions()

    def resample_regressions(self):
        for n, reg in enumerate(self.regressions):
            reg.resample([(X, Y[:,n]) for (X,Y) in self.data_list])


# Alias the "GLM"
GLM = NonlinearAutoregressiveModel

# Define default GLMs for various regression classes
class _DefaultMixin(object):
    _regression_class = None
    def __init__(self, N, B=10, basis=None, **kwargs):
        B = B if basis is None else basis.shape[1]
        regressions = [self._regression_class(N, B, **kwargs) for _ in range(N)]
        super(_DefaultMixin, self).__init__(N, regressions, B=B, basis=basis)

class GaussianGLM(_DefaultMixin, GLM):
    _regression_class = pyglm.regression.GaussianRegression

class SparseGaussianGLM(_DefaultMixin, GLM):
    _regression_class = pyglm.regression.SparseGaussianRegression

class BernoulliGLM(_DefaultMixin, GLM):
    _regression_class = pyglm.regression.BernoulliRegression

class SparseBernoulliGLM(_DefaultMixin, GLM):
    _regression_class = pyglm.regression.SparseBernoulliRegression