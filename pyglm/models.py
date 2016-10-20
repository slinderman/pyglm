import numpy as np
from pybasicbayes.abstractions import Model

import pyglm.regression
from pyglm.utils.basis import convolve_with_basis

class NonlinearAutoregressiveModel(Model):
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
             data_index=0,
             N_to_plot=2):
        """
        Plot the parameters of the model
        :return:
        """
        N, W, A, means = self.N, self.weights, self.adjacency, self.means[0]

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if handles is None:
            handles = []

            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(N_to_plot,3)
            W_ax = fig.add_subplot(gs[:, 0])
            A_ax = fig.add_subplot(gs[:, 1])
            lam_axs = [fig.add_subplot(gs[i,2]) for i in range(N_to_plot)]
            axs = (W_ax, A_ax, lam_axs)

            # Plot the weights
            h_W = W_ax.imshow(W[:,:,0], vmin=-W_lim, vmax=W_lim, cmap="RdBu", interpolation="nearest")
            W_ax.set_xlabel("pre")
            W_ax.set_ylabel("post")
            W_ax.set_xticks(np.arange(self.N))
            W_ax.set_xticklabels(np.arange(self.N)+1)
            W_ax.set_yticks(np.arange(self.N))
            W_ax.set_yticklabels(np.arange(self.N)+1)
            W_ax.set_title("Weights")

            divider = make_axes_locatable(W_ax)
            cbax = divider.new_horizontal(size="5%", pad=0.05)
            fig.add_axes(cbax)
            plt.colorbar(h_W, cax=cbax)
            handles.append(h_W)

            h_A = A_ax.imshow(A, vmin=0, vmax=1, cmap="Greys", interpolation="nearest")
            A_ax.set_xlabel("pre")
            A_ax.set_ylabel("post")
            A_ax.set_title("Adjacency")
            A_ax.set_xticks(np.arange(self.N))
            A_ax.set_xticklabels(np.arange(self.N) + 1)
            A_ax.set_yticks(np.arange(self.N))
            A_ax.set_yticklabels(np.arange(self.N) + 1)

            handles.append(h_A)

            # Plot the true and inferred rates
            for n in range(min(N, N_to_plot)):
                Y = self.data_list[data_index][1]
                tn = np.where(Y[pltslice, n])[0]
                lam_axs[n].plot(tn, np.ones_like(tn), 'ko', markersize=4)
                h_fr = lam_axs[n].plot(means[pltslice, n], label="True")[0]
                lam_axs[n].set_ylim(-0.05, 1.1)
                lam_axs[n].set_ylabel("$\lambda_{}(t)$".format(n+1))

                if n == 0:
                    lam_axs[n].set_title("Firing Rates")

                if n == min(N, N_to_plot) - 1:
                    lam_axs[n].set_xlabel("Time")
                handles.append(h_fr)

            if title is not None:
                handles.append(fig.suptitle(title))

            plt.tight_layout()

        else:
            # If we are given handles, update the data
            handles[0].set_data(W[:,:,0])
            handles[1].set_data(A)
            for n in range(min(N, N_to_plot)):
                handles[2+n].set_data(np.arange(pltslice.start, pltslice.stop), means[pltslice, n])

            if title is not None:
                handles[-1].set_text(title)
            plt.pause(0.001)

        return fig, axs, handles


# Alias the "GLM"
GLM = NonlinearAutoregressiveModel

# Define default GLMs for various regression classes
class _DefaultMixin(object):
    _regression_class = None
    def __init__(self, N, B=10, basis=None, **kwargs):
        """
        :param N:             Observation dimension.
        :param basis:         Basis onto which the preceding activity is projected.
                              In the "identity" case, this is just a lag matrix
        :param B:             Basis dimensionality.
                              In the "identity" case, this is the number of lags.
        :param kwargs:        arguments to the corresponding regression constructor.
        """
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