"""
Models for neural spike trains.
"""

import abc
import copy

import numpy as np

from pyglm.deps.pybasicbayes.abstractions import Model, ModelGibbsSampling, ModelMeanField

from pyglmdos.internals.observations import BernoulliObservations
from pyglmdos.internals.activation import DeterministicActivation
from pyglmdos.internals.bias import GaussianBias
from pyglmdos.internals.background import NoBackground
from pyglmdos.internals.weights import SpikeAndSlabGaussianWeights
from pyglmdos.internals.networks import GibbsSBM

from pyglmdos.utils.basis import CosineBasis

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

    _network_class              = GibbsSBM
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

    def augment_data(self, S):
        T = S.shape[0]
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
        #
        # # Filter the data into a TxKxB array
        # if F is not None:
        #     assert isinstance(F, np.ndarray) and F.shape == (T, self.N, self.B), \
        #         "F must be a filtered event count matrix"
        # else:
        #     print "Convolving with basis"
        #     F = self.basis.convolve_with_basis(S)
        #
        # Add the data in minibatches
        if minibatchsize is None:
            minibatchsize = T

        for offset in np.arange(T, step=minibatchsize):
            end = min(offset+minibatchsize, T)
            S_mb = S[offset:end, :]
            # T_mb = end - offset
            # F_mb = F[offset:end, :]

            # Augment the data with extra local variables and regressors
            # augmented_data = {"T": T_mb, "S": S_mb, "F": F_mb}
            #
            # # The model components may require local variables for each data point
            # self.observation_model.augment_data(augmented_data)
            # self.activation_model.augment_data(augmented_data)
            # self.background_model.augment_data(augmented_data)
            # self.weight_model.augment_data(augmented_data)

            augmented_data = self.augment_data(S_mb)

            # Add to the data list
            self.data_list.append(augmented_data)

    def generate(self, keep=True, T=100, X_bkgd=None, verbose=True):
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

        # If we have some features to start with, add them now
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
            X[t+1:t+L+1,:] += dX

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

    def log_probability(self):
        pass

    def heldout_log_likelihood(self, S):
        pass

    def compute_rate(self, augmented_data):
        # Compute the activation
        Psi = self.activation_model.compute_psi(augmented_data)

        # Compute the expected spike count
        ES = self.observation_model.expected_S(Psi)

        # Normalize by the bin size
        return ES / self.dt


class _GibbsPopulation(_BayesianPopulationBase, ModelGibbsSampling):
    """
    Implement Gibbs sampling for the population model
    """
    def resample_model(self):
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
        raise NotImplementedError()

    def get_vlb(self):
        raise NotImplementedError()


class _SVIPopulation(_MeanFieldPopulation):
    """
    Implement stochastic variational inference for the population model
    """
    def svi_step(self):
        raise NotImplementedError()