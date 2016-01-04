"""
Models for neural spike trains.
"""

import abc
import copy
import sys

import numpy as np
from scipy.misc import logsumexp

from pybasicbayes.abstractions import Model, ModelGibbsSampling, ModelMeanField, ModelParallelTempering
from pybasicbayes.util.text import progprint_xrange

from graphistician.networks import GaussianBernoulliNetwork

from pyglm.internals.observations import BernoulliObservations, NegativeBinomialObservations
from pyglm.internals.activation import DeterministicActivation
from pyglm.internals.bias import GaussianBias
from pyglm.internals.background import NoBackground, LinearDynamicalSystemBackground
from pyglm.internals.weights import NoWeights, SpikeAndSlabGaussianWeights

from pyglm.utils.basis import CosineBasis
from pyglm.utils.profiling import line_profiled
PROFILING = False


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

    _network_class              = GaussianBernoulliNetwork
    _default_network_hypers     = {}


    def __init__(self, N, dt=1.0, dt_max=10.0, B=5,
                 basis=None, basis_hypers={},
                 observation=None, observation_hypers={},
                 activation=None, activation_hypers={},
                 bias=None, bias_hypers={},
                 background=None, background_hypers={},
                 weights=None, weight_hypers={},
                 network=None, network_hypers={},
                 standard_model=None):
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
            self.network = self._network_class(self.N, self.B,
                                               **self.network_hypers)

        # Check that the model doesn't allow instantaneous self connections
        assert not (self.basis.allow_instantaneous and
                    self.network.allow_self_connections), \
            "Cannot allow instantaneous self connections"

        # Initialize the weight model
        if weights is not None:
            assert weights.N == self.N
            self.weight_model = weights
        else:
            self.weight_hypers = copy.deepcopy(self._default_weight_hypers)
            self.weight_hypers.update(weight_hypers)
            self.weight_model = self._weight_class(self, **self.weight_hypers)

        # Initialize the data list to empty
        self.data_list = []

        if standard_model:
            self.initialize_with_standard_model(standard_model)

    def initialize_from_prior(self):
        # This is not equivalent to REsampling with temperature=0!
        self.network.initialize_from_prior()
        self.bias_model.initialize_from_prior()
        self.weight_model.initialize_from_prior()
        self.activation_model.initialize_from_prior()

    def initialize_with_standard_model(self, standard_model):
        """
        Initialize the model parameters with a standard model.
        :param standard_model:
        :return:
        """
        assert standard_model.B == self.B
        self.basis = copy.deepcopy(standard_model.basis)
        self.weight_model.initialize_with_standard_model(standard_model)
        self.bias_model.initialize_with_standard_model(standard_model)
        self.network.initialize_hypers(None, standard_model.W)

    def initialize_with_model(self, standard_model):
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

        F_flat = F.reshape((T,self.N*self.B))
        F_full = np.concatenate((np.ones((T,1)), F_flat), axis=1)

        # Augment the data with extra local variables and regressors
        augmented_data = {"T": T, "S": S, "F": F, "F_full": F_full}

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

    def generate(self, keep=True, T=100, return_Psi=False, verbose=True):
        """
        Generate data from the model.

        :param keep:    Add the data to the model's datalist
        :param T:       Number of time bins to simulate
        :return:
        """
        if T == 0:
            return np.zeros((0,self.N))

        N = self.N
        assert isinstance(T, int), "Size must be an integer number of time bins"
        B = self.basis.B    # Number of features per spike train
        L = self.basis.L    # Length of the impulse responses

        # Initialize output matrix of spike counts
        S = np.zeros((T,N))
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

        # Simulate the background
        X += self.background_model.generate(T+L)

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

            # if np.any(S[t,:]>100):
            #     print "More than 10 spikes in a bin! " \
            #           "Decrease variance on impulse weights or decrease simulation bin width."
            #     import pdb; pdb.set_trace()

        # Cast S to int32
        assert np.all(np.isfinite(S[t,:]))
        assert np.amin(S) >= 0
        assert np.amax(S) <= 1e5
        S = S.astype(np.int32)

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
        ll += self.observation_model.log_likelihood(augmented_data).sum()
        ll += self.activation_model.log_likelihood(augmented_data)
        # ll += self.weight_model.log_likelihood(augmented_data)
        return ll
        
    def log_probability(self, temperature=1.0):
        """
        Compute the log probability of the datasets
        """
        assert temperature >= 0.0 and temperature <= 1.0

        lp = self.log_prior()
        for augmented_data in self.data_list:
            lp += self.log_likelihood(augmented_data).sum() * temperature

        return lp

    def heldout_log_likelihood(self, S, F=None):
        """
        Compute the heldout log likelihood on a spike train, S.
        """
        self.add_data(S, F=F)
        hll = self.log_likelihood(self.data_list[-1]).sum()
        self.data_list.pop()
        return hll

    def heldout_neuron_log_likelihood(self, Strain, Stest, M=100):
        """
        Approximate the log likelihood of a heldout spike train, Stest,
        given the activity of neurons that were observed during training,
        Strain. We assume that the weights from the test
        neuron to the training neuron are all zero such that the test
        neuron does not affect our estimates of the weights of the training
        neurons. To estimate the pred ll, we sample latent variables associated
        with the new neuron and use those to fill in a new column of the weight
        matrix, then we use the updated weight matrix to compute the activation
        for the new neuron. Finally, we use the observation model to compute
        the likelihood of Stest.
        """
        assert Strain.ndim == 2 and Strain.shape[1] == self.N
        T = Strain.shape[0]
        assert Stest.shape == (T,)

        # Compute the filtered spike train
        Sfull = np.hstack((Strain, Stest[:,None]))
        Ffull = self.basis.convolve_with_basis(Sfull)

        plls = []
        for m in xrange(M):
            # Sample a new column of the weight matrix for the new neuron
            Arow, Acol, Wrow, Wcol = self.network.sample_predictive_distribution()
            Wnew = Wcol * Acol[:,None]

            # Sample a new bias for the new neuron
            bnew = self.bias_model.sample_predictive_distribution()

            # Compute the new activation
            psi = np.zeros((T,))
            if not np.allclose(Wnew,0):
                np.einsum("tmb,mb->t", Ffull, Wnew, out=psi)
            psi += bnew

            # Use the observation model to compute the held out likelihood
            pll = self.observation_model.\
                _log_likelihood_given_activation(Stest, psi).sum()
            plls.append(pll)

        # Take the average of the predictive log likelihoods
        return -np.log(M) + logsumexp(plls)

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
    def initialize_with_standard_model(self, standard_model):
        super(_GibbsPopulation, self).\
            initialize_with_standard_model(standard_model)

        # Update the network model a few times
        N_network_updates = 10
        for itr in xrange(N_network_updates):
            self.network.resample((self.weight_model.A, self.weight_model.W))

    def resample_model(self, temperature=1.0):
        assert temperature >= 0.0 and temperature <= 1.0

        # update model components one at a time
        self.observation_model.resample(self.data_list, temperature=temperature)
        self.activation_model.resample(self.data_list)
        self.weight_model.resample(self.data_list)
        self.bias_model.resample(self.data_list)
        self.background_model.resample(self.data_list)

        # Resample the network given the weight model
        self.network.resample((self.weight_model.A, self.weight_model.W))

    @line_profiled
    def collapsed_resample_model(self, temperature=1.0):
        assert temperature >= 0.0 and temperature <= 1.0

        # update model components one at a time
        self.observation_model.resample(self.data_list, temperature=temperature)
        self.activation_model.resample(self.data_list)
        self.weight_model.collapsed_resample(self.data_list)
        self.bias_model.collapsed_resample(self.data_list)
        self.background_model.resample(self.data_list)

        # Resample the network given the weight model
        self.network.resample((self.weight_model.A, self.weight_model.W))

    def ais(self, N_samples=100, B=1000, steps_per_B=1,
            verbose=True, full_output=False, callback=None):
        """
        Since Gibbs sampling as a function of temperature is implemented,
        we can use AIS to approximate the marginal likelihood of the model.
        """
        # We use a linear schedule by default
        betas = np.linspace(0, 1, B)

        print "Estimating marginal likelihood with AIS"
        lw = np.zeros(N_samples)
        for m in progprint_xrange(N_samples):
            # Initialize the model with a draw from the prior
            self.initialize_from_prior()

            # Keep track of the log of the m-th weight
            # It starts at zero because the prior is assumed to be normalized
            lw[m] = 0.0

            # Sample the intermediate distributions
            for b in xrange(1,B):
                if verbose:
                    sys.stdout.write("M: %d\tBeta: %.3f \r" % (m,betas[b]))
                    sys.stdout.flush()

                # Compute the ratio of this sample under this distribution
                # and the previous distribution. The difference is added
                # to the log weight
                curr_lp = self.log_probability(temperature=betas[b])
                prev_lp = self.log_probability(temperature=betas[b-1])
                lw[m] += curr_lp - prev_lp

                # Sample the model at temperature betas[b]
                # Take some number of steps per beta in hopes that
                # the Markov chain will reach equilibrium.
                for s in range(steps_per_B):
                    self.collapsed_resample_model(temperature=betas[b])

                # Call the given callback
                if callback:
                    callback(self, m, b)

            if verbose:
                print ""
                print "W: %f" % lw[m]


        # Compute the mean of the weights to get an estimate of the normalization constant
        log_Z = -np.log(N_samples) + logsumexp(lw)

        # Use bootstrap to compute standard error
        subsamples = np.random.choice(lw, size=(100, N_samples), replace=True)
        log_Z_subsamples = logsumexp(subsamples, axis=1) - np.log(N_samples)
        std_log_Z = log_Z_subsamples.std()

        if full_output:
            return log_Z, std_log_Z, lw
        else:
            return log_Z, std_log_Z


class _MeanFieldPopulation(_BayesianPopulationBase, ModelMeanField):
    """
    Implement mean field variational inference for the population model
    """
    def initialize_with_standard_model(self, standard_model):
        super(_MeanFieldPopulation, self).\
            initialize_with_standard_model(standard_model)

        # Update the network model a few times
        # print "Mean field initializing network:"
        N_network_updates = 10
        for itr in xrange(N_network_updates):
            # sys.stdout.write(".")
            # sys.stdout.flush()
            self.network.meanfieldupdate(self.weight_model)

    def meanfield_coordinate_descent_step(self):
        # TODO: Support multiple datasets
        assert len(self.data_list) == 1, "Can only do mean field variational inference with one dataset"
        data = self.data_list[0]

        # update model components one at a time
        self.observation_model.meanfieldupdate(data)
        self.activation_model.meanfieldupdate(data)
        self.weight_model.meanfieldupdate(data)
        self.bias_model.meanfieldupdate(data)

        # Update the network given the weights
        self.network.meanfieldupdate(self.weight_model)

    def get_vlb(self):
        # TODO: Support multiple datasets
        assert len(self.data_list) == 1, "Can only compute VLBs with one dataset"
        data = self.data_list[0]

        vlb = 0
        vlb += self.observation_model.get_vlb(data)
        vlb += self.activation_model.get_vlb(data)
        vlb += self.weight_model.get_vlb(data)
        vlb += self.bias_model.get_vlb(data)
        vlb += self.network.get_vlb()

        return vlb

    def resample_from_mf(self):
        # TODO: Support multiple datasets
        # assert len(self.data_list) == 1, "Can only compute VLBs with one dataset"
        data = self.data_list[0]

        self.observation_model.resample_from_mf(data)
        self.activation_model.resample_from_mf(data)
        self.weight_model.resample_from_mf(data)
        self.bias_model.resample_from_mf(data)
        self.network.resample_from_mf()

class _SVIPopulation(_BayesianPopulationBase):
    """
    Implement stochastic variational inference for the population model
    """
    def svi_step(self, stepsize):
        # Randomly select a minibatch from the data_list
        mb = self.data_list[np.random.choice(len(self.data_list))]

        # Compute the fraction of the total data this minibatch represents
        mbfrac = float(mb["T"]) / np.sum([d["T"] for d in self.data_list])

        # update model components one at a time
        self.observation_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        self.activation_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        self.weight_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)
        self.bias_model.svi_step(mb, minibatchfrac=mbfrac, stepsize=stepsize)

        # Update the network given the weight model
        self.network.svi_step(self.weight_model, minibatchfrac=mbfrac, stepsize=stepsize)

class _ParallelTemperingPopulation(_BayesianPopulationBase, ModelParallelTempering):
    """
    Implement Parallel Tempring for the population model
    """
    @property
    def temperature(self):
        if not hasattr(self, "_temperature"):
            self._temperature = 1.0
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def energy(self):
        return -self.log_probability(temperature=self.temperature)

    def swap_sample_with(self,other):
        """
        Swap the internals of the this model with the other model
        :param other:
        :return:
        """
        pass


class Population(_GibbsPopulation, _MeanFieldPopulation, _SVIPopulation):
    """
    The default population has a Bernoulli observation model
    and an Erdos-Renyi network.
    """
    pass


class NegativeBinomialPopulation(_GibbsPopulation, _MeanFieldPopulation, _SVIPopulation):
    """
    Population with negative binomial observations and Erdos-Renyi network
    """
    _observation_class          = NegativeBinomialObservations
    _default_observation_hypers = {"xi": 10.0}


class NegativeBinomialEmptyPopulation(NegativeBinomialPopulation):
    """
    Population with negative binomial observations and Erdos-Renyi network
    """
    _weight_class               = NoWeights
    _default_weight_hypers      = {}


# class BernoulliEigenmodelPopulation(Population):
#     _network_class              = GaussianWeightedEigenmodel
#     _default_network_hypers     = {"D": 2, "p": 0.05, "sigma_F": 10, "lmbda": 1*np.ones(2)}
#
#
# class NegativeBinomialEigenmodelPopulation(NegativeBinomialPopulation):
#     _network_class              = GaussianWeightedEigenmodel
#     _default_network_hypers     = {"D": 2, "p": 0.01, "sigma_F": 2**2, "lmbda": 1*np.ones(2)}
#
#
# class BernoulliDistancePopulation(_GibbsPopulation):
#     _network_class              = GaussianDistanceModel
#     _default_network_hypers     = {"D": 2}
#
#
# class NegativeBinomialDistancePopulation(_GibbsPopulation):
#     _network_class              = GaussianDistanceModel
#     _default_network_hypers     = {"D": 2}
#
#     _observation_class          = NegativeBinomialObservations
#     _default_observation_hypers = {"xi": 10.0}


# class ExcitatoryNegativeBinomialDistancePopulation(NegativeBinomialDistancePopulation):
#     # Weight and network class must be specified by subclasses
#     _weight_class               = SpikeAndSlabTruncatedGaussianWeights
#     _default_weight_hypers      = {"lb": 0, "ub": np.inf}
#
#     _network_class              = FixedGaussianDistanceModel
#     _default_network_hypers     = {"D": 2}
#
#     _observation_class          = NegativeBinomialObservations
#     _default_observation_hypers = {"xi": 10.0}
#
#
# class NegativeBinomialLDSPopulation(NegativeBinomialPopulation):
#     # Weight and network class must be specified by subclasses
#     _weight_class               = NoWeights
#     _default_weight_hypers      = {}
#
#     _background_class           = LinearDynamicalSystemBackground
#     _default_background_hypers  = {"D": 2}
#
#     _observation_class          = NegativeBinomialObservations
#     _default_observation_hypers = {"xi": 10.0}
#
#     def heldout_log_likelihood(self, S, F=None, n_resamples=100):
#         """
#         Compute the heldout log likelihood on a spike train, S.
#         """
#         self.add_data(S, F=F)
#         data = self.data_list[-1]
#         # We need to integrate out the latent states of the LDS
#         hll_smpls = np.zeros(n_resamples)
#         for itr in xrange(n_resamples):
#             data["states"] = self.background_model.generate_states(data["T"])
#             hll_smpls[itr] = self.log_likelihood(data).sum()
#
#         # Compute the expectation in log space
#         from scipy.misc import logsumexp
#         hll = -np.log(n_resamples) + logsumexp(hll_smpls)
#
#         # Remove the data
#         self.data_list.pop()
#
#         return hll

# class BernoulliSBMPopulation(Population):
#     _network_class              = GaussianStochasticBlockModel
#     _default_network_hypers     = {"C": 2, "p": 0.25}
#
#
# class NegativeBinomialSBMPopulation(NegativeBinomialPopulation):
#     _network_class              = GaussianStochasticBlockModel
#     _default_network_hypers     = {"C": 2}
