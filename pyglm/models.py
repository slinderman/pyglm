import numpy as np
from pybasicbayes.abstractions import ModelGibbsSampling

import pyglm.networks
import pyglm.regression
from pyglm.utils.basis import convolve_with_basis

class NonlinearAutoregressiveModel(ModelGibbsSampling):
    """
    The "generalized linear model" in neuroscience is really
    a vector autoregressive model. As the name suggests,
    the key component in these models is a regression from
    inputs, x, to outputs, y.

    When the outputs are discrete random variables, like spike
    counts, we typically take the regression to be a generalized
    linear model:

       y ~ p(mu(x), theta)
       mu(x) = f(w \dot x)

    where 'p' is a discrete distribution, like the Poisson,
    and 'f' is a "link" function that maps a linear function of
    x to the parameters of 'p'. Hence the name "GLM" in
    computational neuroscience.
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
        return np.array([r.W for r in self.regressions])

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

    def generate(self, keep=True, T=100, verbose=False, intvl=10):
        """
        Generate data from the model.

        :param keep:    Add the data to the model's datalist
        :param T:       Number of time bins to simulate
        :param verbose: Whether or not to print status
        :param intvl:   Number of intervals between printing status

        :return X:      Convolution of data with basis functions
        :return Y:      Generate data matrix
        """
        if T == 0:
            return np.zeros((0,self.N))
        assert isinstance(T, int), "Size must be an integer number of time bins"

        N, basis = self.N, self.basis
        L, B = basis.shape

        # NOTE: the basis is defined such that the first row is the
        #       previous time step and the last row is T-L steps in
        #       the past. Thus, for the dot products below, we need
        #       to flip the basis matrix.
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
                if t % intvl == 0:
                    print("Generate t={}".format(t))
            # 1. Project previous activity window onto the basis
            #    previous activity is L x N, basis is L x B,
            X[t] = Y[t-L:t].T.dot(basis)

            # 2. Compute the activation, W.dot(X[t]) + b
            Psi[t] = W.dot(X[t].reshape((N*B,))) + b

            # 3. Weight the previous activity with the regression weights
            Y[t] = self.regressions[0].rvs(psi=Psi[t])

        if keep:
            self.add_data(Y[L:], X=X[L:])

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

    ### Plotting
    def plot(self,
             fig=None,
             axs=None,
             handles=None,
             title=None,
             figsize=(6,3),
             W_lim=3,
             pltslice=slice(0, 500),
             N_to_plot=2,
             data_index=0):
        """
        Plot the parameters of the model
        :return:
        """
        from pyglm.plotting import plot_glm
        return plot_glm(
            self.data_list[data_index][1],
            self.weights,
            self.adjacency,
            self.means[0],
            fig=fig,
            axs=axs,
            handles=handles,
            title=title,
            figsize=figsize,
            W_lim=W_lim,
            pltslice=pltslice,
            N_to_plot=N_to_plot)

class HierarchicalNonlinearAutoregressiveModel(NonlinearAutoregressiveModel):
    """
    The network GLM is really just a hierarchical AR model. We specify a
    prior distribution on the weights of a collection of conditionally
    independent AR models. Since these weights are naturally interpreted
    as a network, we refer to these as "network" AR models, or "network GLMs".
    """

    def __init__(self, N, network, regressions, basis=None, B=10):
        """
        The only difference here is that we also provide a 'network' object,
        which specifies a prior distribution on the regression weights.

        :param network:
        """
        super(HierarchicalNonlinearAutoregressiveModel, self). \
            __init__(N, regressions, basis=basis, B=B)

        self.network = network

    def resample_model(self):
        super(HierarchicalNonlinearAutoregressiveModel, self).resample_model()
        self.resample_network()

    def resample_network(self):
        net = self.network
        net.resample((self.adjacency, self.weights))

        # Update the regression hyperparameters
        for n, reg in enumerate(self.regressions):
            reg.S_w = net.sigma_W[n]
            reg.mu_w = net.mu_W[n]
            reg.rho = net.rho[n]

# Alias the "GLM"
GLM = NonlinearAutoregressiveModel
NetworkGLM = HierarchicalNonlinearAutoregressiveModel

# Define default GLMs for various regression classes
class _DefaultMixin(object):
    _network_class = None
    _regression_class = None
    def __init__(self, N, B=10, basis=None,
                 network=None,
                 network_kwargs=None,
                 regressions=None,
                 regression_kwargs=None):
        """
        :param N:             Observation dimension.
        :param basis:         Basis onto which the preceding activity is projected.
                              In the "identity" case, this is just a lag matrix
        :param B:             Basis dimensionality.
                              In the "identity" case, this is the number of lags.
        :param kwargs:        arguments to the corresponding regression constructor.
        """
        B = B if basis is None else basis.shape[1]
        if network is None:
            network_kwargs = dict() if network_kwargs is None else network_kwargs
            network = self._network_class(N, B, **network_kwargs)

        if regressions is None:
            regression_kwargs = dict() if regression_kwargs is None else regression_kwargs
            regressions = [self._regression_class(N, B, **regression_kwargs) for _ in range(N)]
        super(_DefaultMixin, self).__init__(N, network, regressions, B=B, basis=basis)


class GaussianGLM(_DefaultMixin, NetworkGLM):
    _network_class = pyglm.networks.NIWDenseNetwork
    _regression_class = pyglm.regression.GaussianRegression

class SparseGaussianGLM(_DefaultMixin, NetworkGLM):
    _network_class = pyglm.networks.NIWFixedSparsityNetwork
    _regression_class = pyglm.regression.SparseGaussianRegression

class BernoulliGLM(_DefaultMixin, NetworkGLM):
    _network_class = pyglm.networks.NIWDenseNetwork
    _regression_class = pyglm.regression.BernoulliRegression

class SparseBernoulliGLM(_DefaultMixin, NetworkGLM):
    _network_class = pyglm.networks.NIWFixedSparsityNetwork
    _regression_class = pyglm.regression.SparseBernoulliRegression
