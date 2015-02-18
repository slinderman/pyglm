"""
Models for neural spike trains.
"""

import abc
import copy
import sys

import numpy as np

from pyglm.deps.pybasicbayes.abstractions import Model, ModelGibbsSampling, ModelMeanField

from pyglm.internals.observations import BernoulliObservations
from pyglm.internals.activation import DeterministicActivation
from pyglm.internals.bias import GaussianBias
from pyglm.internals.background import NoBackground
from pyglm.internals.weights import SpikeAndSlabGaussianWeights
from pyglm.internals.networks import StochasticBlockModel

from pyglm.utils.basis import CosineBasis
from pyglm.utils.utils import logistic

class StandardBernoulliPopulation(Model):

    # Define the model components and their default hyperparameters
    _basis_class                = CosineBasis
    _default_basis_hypers       = {'norm': True, 'allow_instantaneous': False}


    def __init__(self, N, dt=1.0, dt_max=10.0, B=5,
                 basis=None, basis_hypers={},
                 allow_self_connections=True):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param N:  Number of processes
        """
        self.N      = N
        self.dt     = dt
        self.dt_max = dt_max
        self.B      = B
        self.allow_self_connections = allow_self_connections

        # Initialize the basis
        if basis is not None:
            # assert basis.B == B
            self.basis = basis
            self.B     = basis.B
        else:
            # Use the given basis hyperparameters
            self.basis_hypers = copy.deepcopy(self._default_basis_hypers)
            self.basis_hypers.update(basis_hypers)
            self.basis = self._basis_class(self.B, self.dt, self.dt_max,
                                           **self.basis_hypers)

        # Initialize the weights of the standard model.
        # We have a weight for the background
        self.b = np.zeros(self.N)
        # And a weight for each basis function of each presynaptic neuron.
        self.weights = 1e-3 * np.ones((self.N, self.N*self.B))
        if not self.allow_self_connections:
            self._remove_self_weights()

        # Initialize the data list to empty
        self.data_list = []

    @property
    def W(self):
        WB = self.weights.reshape((self.N,self.N, self.B))

        # DEBUG
        assert WB[0,0,self.B-1] == self.weights[0,self.B-1]
        assert WB[0,self.N-1,0] == self.weights[0,(self.N-1)*self.B]

        if self.B > 2:
            assert WB[self.N-1,self.N-1,self.B-2] == self.weights[self.N-1,-2]

        # Weight matrix is summed over impulse response functions
        W = np.transpose(WB, axes=[1,0,2])

        return W

    @property
    def bias(self):
        return self.b

    def _remove_self_weights(self):
        for n in xrange(self.N):
                self.weights[n,(n*self.B):(n+1)*self.B] = 1e-32


    def augment_data(self, S):
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.N \
               and np.amin(S) >= 0 and S.dtype == np.int32, \
               "Data must be a TxN array of event counts"

        T = S.shape[0]

        # Filter the data into a TxKxB array
        Ftens = self.basis.convolve_with_basis(S)

        # Flatten this into a T x (KxB) matrix
        # [F00, F01, F02, F10, F11, ... F(K-1)0, F(K-1)(B-1)]
        F = Ftens.reshape((T, self.N * self.B))
        assert np.allclose(F[:,0], Ftens[:,0,0])
        if self.B > 1:
            assert np.allclose(F[:,1], Ftens[:,0,1])
        if self.N > 1:
            assert np.allclose(F[:,self.B], Ftens[:,1,0])

        augmented_data = {"T": T, "S": S, "F": F}
        return augmented_data


    def add_data(self, S, F=None, minibatchsize=None):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: a TxN matrix of of event counts for each time bin
                  and each neuron.
        """
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.N \
               and np.amin(S) >= 0 and S.dtype == np.int32, \
               "Data must be a TxN array of event counts"

        T = S.shape[0]

        if minibatchsize is None:
            minibatchsize = T

        for offset in np.arange(T, step=minibatchsize):
            end = min(offset+minibatchsize, T)
            S_mb = S[offset:end,:]

            augmented_data = self.augment_data(S_mb)

            # Add minibatch to the data list
            self.data_list.append(augmented_data)

    def copy_sample(self):
        """
        Return a copy of the parameters of the model
        :return: The parameters of the model (A,W,\lambda_0, \beta)
        """
        # return copy.deepcopy(self.get_parameters())

        # Shallow copy the data
        data_list = copy.copy(self.data_list)
        self.data_list = []

        # Make a deep copy without the data
        model_copy = copy.deepcopy(self)

        # Reset the data and return the data-less copy
        self.data_list = data_list
        return model_copy

    def generate(self,keep=True,**kwargs):
        raise NotImplementedError()

    def compute_rate(self, augmented_data):
        """
        Compute the rate of the augmented data

        :param index:   Which dataset to compute the rate of
        :param ns:      Which neurons to compute the rate of
        :return:
        """
        F = augmented_data["F"]
        R = np.zeros((augmented_data["T"], self.N))
        for n in xrange(self.N):
            Xn = F.dot(self.weights[n,:])
            Xn += self.bias[n]
            R[:,n] = logistic(Xn)

        return R

    def log_likelihood(self, augmented_data=None):
        """
        Compute the log likelihood of the augmented data
        :return:
        """
        ll = 0

        if augmented_data is None:
            datas = self.data_list
        else:
            datas = [augmented_data]

        ll = 0
        for data in datas:
            S = data["S"]
            R = self.compute_rate(data)
            ll += (S * np.log(R) + (1-S) * np.log(1-R)).sum()

        return ll

    def heldout_log_likelihood(self, S=None, augmented_data=None):
        if S is not None and augmented_data is None:
            augmented_data = self.augment_data(S)
        elif S is None and augmented_data is None:
            raise Exception("Either S or augmented data must be given")

        return self.log_likelihood(augmented_data)


    def fit(self, L1=True):
        """
        Use scikit-learn's LogisticRegression model to fit the data

        :param L1:  If True, use L1 penalty on the coefficients
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import l1_min_c

        print "Initializing with logistic regresion"
        F = np.vstack([d["F"] for d in self.data_list])
        S = np.vstack([d["S"] for d in self.data_list])

        if L1:
            # Hold out some data for cross validation
            offset = int(0.75 * S.shape[0])
            T_xv = S.shape[0] - offset
            F_xv = F[offset:, ...]
            S_xv = S[offset:, ...]
            augmented_xv_data = {"T": T_xv, "S": S_xv, "F": F_xv}

            F    = F[:offset, ...]
            S    = S[:offset, ...]

            for n_post in xrange(self.N):
                # Get a L1 regularization path for inverse penalty C
                cs = l1_min_c(F, S[:,n_post], loss='log') * np.logspace(1, 4., 10)
                # The intercept is also subject to penalization, even though
                # we don't really want to penalize it. To counteract this effect,
                # we scale the intercept by a large value
                intercept_scaling = 10**6


                print "Computing regularization path for neuron %d ..." % n_post
                ints      = []
                coeffs    = []
                xv_scores = []
                lr = LogisticRegression(C=1.0, penalty='l1',
                                        fit_intercept=True, intercept_scaling=intercept_scaling,
                                        tol=1e-6)
                for c in cs:
                    lr.set_params(C=c)
                    lr.fit(F, S[:,n_post])
                    ints.append(lr.intercept_.copy())
                    coeffs.append(lr.coef_.ravel().copy())
                    # xv_scores.append(lr.score(F_xv, S_xv[:,n_post]).copy())

                    # Temporarily set the weights and bias
                    self.b[n_post] = lr.intercept_
                    self.weights[n_post, :] = lr.coef_
                    xv_scores.append(self.heldout_log_likelihood(augmented_data=augmented_xv_data))

                # Choose the regularization penalty with cross validation
                print "XV Scores: "
                for c,score  in zip(cs, xv_scores):
                    print "\tc: {%.2f}\tscore: {%.3f}" % (c,score)
                best = np.argmax(xv_scores)
                print "Best c: ", cs[best]

                # Save the best weights
                self.b[n_post]          = ints[best]
                self.weights[n_post, :] = coeffs[best]

        else:
            # Just use standard L2 regularization
            for n_post in xrange(self.N):
                sys.stdout.write('.')
                sys.stdout.flush()

                lr = LogisticRegression(fit_intercept=True)
                lr.fit(F,S[:,n_post])
                self.b[n_post] = lr.intercept_
                self.weights[n_post,:] = lr.coef_

        print ""


class _BayesianPopulationBase(Model):
    """
    Base model for a population of neurons
    """
    __metaclass__ = abc.ABCMeta

    # Define the model components and their default hyperparameters
    _basis_class                = CosineBasis
    _default_basis_hypers       = {'norm': True, 'allow_instantaneous': False}

    _observation_class          = BernoulliObservations
    _default_observation_hypers = {}

    _activation_class           = DeterministicActivation
    _default_activation_hypers  = {}

    _bias_class                 = GaussianBias
    _default_bias_hypers        = {'mu_0': 0.0, 'sigma_0': 1.0}

    _background_class           = NoBackground
    _default_background_hypers  = {}

    # Weight and network class must be specified by subclasses
    _weight_class               = SpikeAndSlabGaussianWeights
    _default_weight_hypers      = {}

    _network_class              = StochasticBlockModel
    _default_network_hypers     = {"C": 1}


    def __init__(self, N, dt=1.0, dt_max=10.0, B=5,
                 basis=None, basis_hypers={},
                 observation=None, observation_hypers={},
                 activation=None, activation_hypers={},
                 bias=None, bias_hypers={},
                 background=None, background_hypers={},
                 weights=None, weight_hypers={},
                 network=None, network_hypers={}):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param N:  Number of processes
        """
        self.N      = N
        self.dt     = dt
        self.dt_max = dt_max
        self.B      = B

        # Initialize the basis
        if basis is not None:
            # assert basis.B == B
            self.basis = basis
            self.B     = basis.B
        else:
            # Use the given basis hyperparameters
            self.basis_hypers = copy.deepcopy(self._default_basis_hypers)
            self.basis_hypers.update(basis_hypers)
            self.basis = self._basis_class(self.B, self.dt, self.dt_max,
                                           **self.basis_hypers)

        # Initialize the observation model
        if observation is not None:
            self.observation_model = observation
        else:
            # Use the given basis hyperparameters
            self.observation_hypers = copy.deepcopy(self._default_observation_hypers)
            self.observation_hypers.update(observation_hypers)
            self.observation_model = self._observation_class(self, **self.observation_hypers)

        # Initialize the activation model
        if activation is not None:
            self.activation_model = activation
        else:
            # Use the given basis hyperparameters
            self.activation_hypers = copy.deepcopy(self._default_activation_hypers)
            self.activation_hypers.update(activation_hypers)
            self.activation_model = self._activation_class(self,  **self.activation_hypers)

        # Initialize the bias
        if bias is not None:
            self.bias_model = bias
        else:
            # Use the given basis hyperparameters
            self.bias_hypers = copy.deepcopy(self._default_bias_hypers)
            self.bias_hypers.update(bias_hypers)
            self.bias_model = self._bias_class(self, **self.bias_hypers)

        # Initialize the background model
        if background is not None:
            self.background_model = background
        else:
            # Use the given background hyperparameters
            self.background_hypers = copy.deepcopy(self._default_background_hypers)
            self.background_hypers.update(background_hypers)
            self.background_model = self._background_class(self, **self.background_hypers)

        # Initialize the network model
        if network is not None:
            self.network = network
        else:
            # Use the given network hyperparameters
            self.network_hypers = copy.deepcopy(self._default_network_hypers)
            self.network_hypers.update(network_hypers)
            self.network = self._network_class(self,
                                               **self.network_hypers)

        # Check that the model doesn't allow instantaneous self connections
        assert not (self.basis.allow_instantaneous and
                    self.network.allow_self_connections), \
            "Cannot allow instantaneous self connections"

        # Initialize the weight model
        if weights is not None:
            assert weights.K == self.N
            self.weight_model = weights
        else:
            self.weight_hypers = copy.deepcopy(self._default_weight_hypers)
            self.weight_hypers.update(weight_hypers)
            self.weight_model = self._weight_class(self, **self.weight_hypers)

        # Initialize the data list to empty
        self.data_list = []

    def initialize_with_standard_model(self, standard_model):
        """
        Initialize the model parameters with a standard model.
        :param standard_model:
        :return:
        """
        self.weight_model.initialize_with_standard_model(standard_model)
        self.bias_model.initialize_with_standard_model(standard_model)

    def augment_data(self, S, F=None):
        T = S.shape[0]

        # Filter the spike train if necessary
        if F is None:
            F = self.basis.convolve_with_basis(S)

        # Augment the data with extra local variables and regressors
        augmented_data = {"T": T, "S": S, "F": F}

        # The model components may require local variables for each data point
        self.observation_model.augment_data(augmented_data)
        self.activation_model.augment_data(augmented_data)
        self.background_model.augment_data(augmented_data)
        self.weight_model.augment_data(augmented_data)

        return augmented_data

    def add_data(self, S, F=None, minibatchsize=None):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: a TxK matrix of of event counts for each time bin
                  and each process.
        """
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.N \
               and np.amin(S) >= 0 and S.dtype == np.int32, \
               "Data must be a TxN array of event counts"

        T = S.shape[0]

        # Add the data in minibatches
        if minibatchsize is None:
            minibatchsize = T

        for offset in np.arange(T, step=minibatchsize):
            end = min(offset+minibatchsize, T)
            S_mb = S[offset:end, :]

            # Extract the filtered spikes if given
            if F is not None:
                F_mb = F[offset:end, ...]
            else:
                F_mb = None

            augmented_data = self.augment_data(S_mb, F=F_mb)

            # Add to the data list
            self.data_list.append(augmented_data)

    def generate(self, keep=True, T=100, X_bkgd=None, return_Psi=False, verbose=True):
        """
        Generate data from the model.

        :param keep:    Add the data to the model's datalist
        :param T:       Number of time bins to simulate
        :return:
        """
        N = self.N
        assert isinstance(T, int), "Size must be an integer number of time bins"
        B = self.basis.B    # Number of features per spike train
        L = self.basis.L    # Length of the impulse responses

        # Initialize output matrix of spike counts
        S = np.zeros((T,N), dtype=np.int32)
        # Initialize the rate matrix
        Psi = np.zeros((T+L,N))
        # TODO: Come up with a better symbol for the activation
        # Initialize the autoregressive activation matrix
        X = np.zeros((T+L, N))

        # Precompute the impulse responses (LxNxN array)
        H = np.tensordot(self.basis.basis, self.weight_model.W, axes=([1], [2]))
        assert H.shape == (L,self.N, self.N)

        # Transpose H so that it is faster for tensor mult
        H = np.transpose(H, axes=[0,2,1])

        # Add the bias
        X += self.bias_model.b[None,:]

        # If we have some background features to start with, add them now
        if X_bkgd is not None:
            T_bkgd = min(T+L, X_bkgd.shape[0])
            for n in range(N):
                assert X_bkgd[n].shape[1] == B
                X[:T_bkgd, n, :] += X_bkgd[n]

        # Cap the number of spikes in a time bin
        max_spks_per_bin = 10
        n_exceptions = 0

        # Iterate over each time step and generate spikes
        if verbose:
            print "Simulating %d time bins" % T

        for t in np.arange(T):
            if verbose:
                if np.mod(t,10000)==0:
                    print "t=%d" % t

            # Compute the rate for the given activation
            Psi[t,:] = self.activation_model.rvs(X[t,:])

            # Sample spike counts
            S[t,:] = self.observation_model.rvs(Psi[t,:])

            # Compute change in activation via tensor product
            dX = np.tensordot( H, S[t,:], axes=([2, 0]))
            X[t:t+L,:] += dX

            # Check Spike limit
            if np.any(S[t,:] >= max_spks_per_bin):
                n_exceptions += 1

            if np.any(S[t,:]>100):
                print "More than 10 spikes in a bin! " \
                      "Decrease variance on impulse weights or decrease simulation bin width."
                import pdb; pdb.set_trace()

        if verbose:
            print "Number of exceptions arising from multiple spikes per bin: %d" % n_exceptions

        if keep:
            # Xs = [X[:T,:] for X in Xs]
            # data = np.hstack(Xs + [S])
            self.add_data(S)

        if return_Psi:
            return S, Psi
        else:
            return S

    def copy_sample(self):
        """
        Return a copy of the parameters of the model
        :return: The parameters of the model (A,W,\lambda_0, \beta)
        """
        # return copy.deepcopy(self.get_parameters())

        # Shallow copy the data
        data_list = copy.copy(self.data_list)
        self.data_list = []

        # Make a deep copy without the data
        model_copy = copy.deepcopy(self)

        # Reset the data and return the data-less copy
        self.data_list = data_list
        return model_copy

    def log_prior(self):
        lprior  = 0
        lprior += self.observation_model.log_prior()
        lprior += self.activation_model.log_prior()
        lprior += self.bias_model.log_prior()
        lprior += self.background_model.log_prior()
        lprior += self.weight_model.log_prior()
        lprior += self.network.log_prior()
    
        return lprior
    
    def log_likelihood(self, augmented_data):
        ll = 0

        return ll
        
    def log_probability(self):
        """
        Compute the log probability of the datasets
        """
        lp = self.log_prior()
        for augmented_data in self.data_list:
            lp += self.log_likelihood(augmented_data).sum()

        return lp

    def heldout_log_likelihood(self, S, F=None):
        """
        Compute the heldout log likelihood on a spike train, S.
        """
        self.add_data(S, F=F)
        self.log_likelihood(self.data_list[-1])
        self.data_list.pop()

    def compute_rate(self, augmented_data):
        # Compute the activation
        Psi = self.activation_model.compute_psi(augmented_data)

        # Compute the expected spike count
        ES = self.observation_model.expected_S(Psi)

        # Normalize by the bin size
        return ES / self.dt

    def mf_expected_rate(self, augmented_data):
        # Compute the activation
        Psi = self.activation_model.mf_expected_activation(augmented_data)

        # Compute the expected spike count
        ES = self.observation_model.expected_S(Psi)

        # Normalize by the bin size
        return ES / self.dt


class _GibbsPopulation(_BayesianPopulationBase, ModelGibbsSampling):
    """
    Implement Gibbs sampling for the population model
    """
    def resample_model(self):
        # TODO: Support multile datasets
        assert len(self.data_list) == 1, "Can only do Gibbs sampling with one dataset"
        data = self.data_list[0]

        # update model components one at a time
        self.observation_model.resample(data)
        self.activation_model.resample(data)
        self.weight_model.resample(data)
        self.bias_model.resample(data)
        # self.network.resample(data)


class _MeanFieldPopulation(_BayesianPopulationBase, ModelMeanField):
    """
    Implement mean field variational inference for the population model
    """
    def meanfield_coordinate_descent_step(self):
        # TODO: Support multile datasets
        assert len(self.data_list) == 1, "Can only do mean field variational inference with one dataset"
        data = self.data_list[0]

        # update model components one at a time
        self.observation_model.meanfieldupdate(data)
        self.activation_model.meanfieldupdate(data)
        self.weight_model.meanfieldupdate(data)
        self.bias_model.meanfieldupdate(data)
        # self.network.meanfieldupdate(data)

    def get_vlb(self):
        # TODO: Support multile datasets
        assert len(self.data_list) == 1, "Can only compute VLBs with one dataset"
        data = self.data_list[0]

        vlb = 0
        vlb += self.observation_model.get_vlb(data)
        vlb += self.activation_model.get_vlb(data)
        vlb += self.weight_model.get_vlb(data)
        vlb += self.bias_model.get_vlb(data)
        # vlb += self.network.get_vlb(data)

        return vlb


class _SVIPopulation(_BayesianPopulationBase):
    """
    Implement stochastic variational inference for the population model
    """
    def svi_step(self, stepsize):
        # Randomly select a minibatch from the data_list
        mb = self.data_list[np.random.choice(len(self.data_list))]

        # Compute the fraction of the total data this minibatch represents
        mbfrac = float(mb["T"]) / sum([d["T"] for d in self.data_list])

        # update model components one at a time
        self.observation_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        self.activation_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        self.weight_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        self.bias_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        # self.network.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)


class Population(_GibbsPopulation, _MeanFieldPopulation, _SVIPopulation):
    """
    The default population has a Bernoulli observation model and an Erdos-Renyi network.
    """
    pass