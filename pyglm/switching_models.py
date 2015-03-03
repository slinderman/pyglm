"""
Create switching population models by using the population model
as an observation distribution in a hidden Markov model.
"""
import numpy as np
import copy

from pyhsmm.models import WeakLimitHDPHMM, WeakLimitHDPHSMM
from pyhsmm.basic.distributions import PoissonDuration
from pyglm.distributions import PopulationDistribution, \
    NegativeBinomialPopulationDistribution, \
    NegativeBinomialEmptyPopulationDistribution

class _HDPHMMPopulationMixin(object):
    """
    A switching population model with Markovian dynamics.
    """
    _population_class = PopulationDistribution

    _default_hdp_hmm_hypers = {"alpha": 6., "gamma": 6.,
                               "init_state_concentration": 6.}

    def __init__(self, N, M,
                 dt=1.0, dt_max=10.0, B=5,                  # Population parameters
                 basis=None, basis_hypers={},
                 observation=None, observation_hypers={},
                 activation=None, activation_hypers={},
                 bias=None, bias_hypers={},
                 background=None, background_hypers={},
                 weights=None, weight_hypers={},
                 network=None, network_hypers={},
                 n_iters_per_resample=10,
                 hdp_hmm_hypers={}                          # HDP-HMM hyperparameters
                 ):
        """
        :param N:   The number of neurons
        :param M:   The maximum number of latent states

        The remainder of the parameters are specified in the Population
        of HMM definitions.
        :return:
        """
        self.N = N
        self.M = M
        self.B = B
        self.dt = dt
        self.dt_max = dt_max

        # Create observation models
        self.population_dists = \
            [self._population_class(N,
                                    dt=dt, dt_max=dt_max, B=B,
                                    basis=basis, basis_hypers=basis_hypers,
                                    observation=observation, observation_hypers=observation_hypers,
                                    activation=activation, activation_hypers=activation_hypers,
                                    bias=bias, bias_hypers=bias_hypers,
                                    background=background, background_hypers=background_hypers,
                                    weights=weights, weight_hypers=weight_hypers,
                                    network=network, network_hypers=network_hypers,
                                    n_iters_per_resample=n_iters_per_resample)
             for m in xrange(M)]

        self.populations = [p.population_model for p in self.population_dists]

        # Initialize the switching model with the populations
        self.hdp_hmm_hypers = copy.deepcopy(self._default_hdp_hmm_hypers)
        self.hdp_hmm_hypers.update(hdp_hmm_hypers)
        super(_HDPHMMPopulationMixin, self).\
            __init__(obs_distns=self.population_dists, **self.hdp_hmm_hypers)

    # TODO: This is pretty hacky...
    @property
    def basis_hypers(self):
        return self.populations[0].basis_hypers

    @property
    def observation_hypers(self):
        return self.populations[0].observation_hypers

    @property
    def activation_hypers(self):
        return self.populations[0].activation_hypers

    @property
    def bias_hypers(self):
        return self.populations[0].bias_hypers

    @property
    def background_hypers(self):
        return self.populations[0].background_hypers

    @property
    def weight_hypers(self):
        return self.populations[0].weight_hypers

    @property
    def network_hypers(self):
        return self.populations[0].network_hypers

    def add_data(self, data, stateseq=None, packed_data=None, **kwargs):
        """
        Add a spike train to the model. We need to augment the spike train
        with its filtered version.

        :param data:        A TxN spike train matrix
        :param stateseq:    The initial state sequence
        :param kwargs:
        :return:
        """
        # Since all the population objects have the same basis, we can
        # filter the spike train with the first population
        if packed_data is None:
            packed_data = self.population_dists[0].pack_spike_train(data)

        super(_HDPHMMPopulationMixin, self).add_data(packed_data, stateseq=stateseq, **kwargs)

        return packed_data

    def pop_data(self):
        return self.states_list.pop()

    def generate(self,T,keep=True, keep_stateseq=False):
        S, stateseq = super(_HDPHMMPopulationMixin, self).generate(T=T, keep=False)

        if keep:
            if keep_stateseq:
                self.add_data(S, stateseq=stateseq)
            else:
                self.add_data(S)

        return S, stateseq

    def compute_rate(self, data_index=0):
        """
        Compute the "firing rate" of the model with its current state sequence.
        :return:
        """
        states = self.states_list[data_index]
        rate = np.empty((states.T, self.N))
        for m in xrange(self.M):
            ts = np.where(states.stateseq==m)
            if len(ts) > 0:
                rate[ts] = self.population_dists[m].compute_rate(states.data[ts])

        return rate

    @property
    def hidden_state_sequence(self):
        hidden_states = [s.stateseq for s in self.states_list]
        return hidden_states

    def heldout_log_likelihood(self, data, packed_data=None):
        """
        Compute heldout log likelihood of a dataset
        """
        self.add_data(data, packed_data=packed_data)
        s = self.states_list.pop()
        return s.log_likelihood()

    # def plot(self, **kwargs):
    #     super(_SwitchingPopulationMixin, self).plot(**kwargs)
    #
    #     import matplotlib.pyplot as plt
    #     fig = plt.gcf()
    #     ax = fig.add_subplot(211)
    #     ax.set_xlim(0, self.states_list[0].T)
    #     ax.set_ylim(0, self.N)
    #
    #     try:
    #         plt.set_cmap("harvard")
    #     except:
    #         pass

    def plot_observations(self,ax=None,color=None,plot_slice=slice(None),update=False):
        import matplotlib.pyplot as plt
        keepaxis = ax is not None
        ax = ax if ax else plt.gca()
        axis = ax.axis()
        #
        # state_colors = super(_SwitchingPopulationMixin, self)._get_colors()
        # usages = super(_SwitchingPopulationMixin, self).state_usages
        # obs_distns = super(_SwitchingPopulationMixin, self).obs_distns
        #
        # artists = []
        # for state, (o, w) in enumerate(zip(obs_distns,usages)):
        #     artists.extend(
        #         o.plot(
        #             color=state_colors[state], label='%d' % state,
        #             alpha=min(0.25,1.-(1.-w)**2)/0.25,
        #             ax=ax, update=update,draw=False))
        #
        # if keepaxis: ax.axis(axis)
        #
        stateseq = self.states_list[0].stateseq[plot_slice]
        t = np.arange(len(stateseq))[plot_slice]
        data = self.states_list[0].data[plot_slice]
        artists = self.population_dists[0].plot(indices=t, data=data, ax=ax)

        ax.set_xlim(0, len(stateseq))
        ax.set_xticklabels([])
        ax.set_ylim(0.5, self.N+.5)
        ax.set_ylabel("Neuron")

        return artists

    def _plot_stateseq_data_values(self,s,ax,state_colors,plot_slice,update):
        return None


class _HDPHSMMPopulationMixin(_HDPHMMPopulationMixin):
    _duration_class = PoissonDuration
    _default_duration_hypers = {'alpha_0':2*30, 'beta_0':2}

    def __init__(self, N, M,
                 dt=1.0, dt_max=10.0, B=5,                  # Population parameters
                 basis=None, basis_hypers={},
                 observation=None, observation_hypers={},
                 activation=None, activation_hypers={},
                 bias=None, bias_hypers={},
                 background=None, background_hypers={},
                 weights=None, weight_hypers={},
                 network=None, network_hypers={},
                 n_iters_per_resample=10,
                 hdp_hmm_hypers={},                          # HDP-HMM hyperparameters
                 duration_hypers={}
                 ):
        """
        :param N:   The number of neurons
        :param M:   The maximum number of latent states

        The remainder of the parameters are specified in the Population
        of HMM definitions.
        :return:
        """
        self.N = N
        self.M = M

        # Initialize duration distributions
        self.duration_hypers = copy.deepcopy(self._default_duration_hypers)
        self.duration_hypers.update(duration_hypers)
        self.dur_distns = [self._duration_class(**self.duration_hypers) for m in range(self.M)]

        hdp_hmm_hypers["dur_distns"] = self.dur_distns

        super(_HDPHSMMPopulationMixin, self).\
            __init__(N, M, dt=dt, dt_max=dt_max, B=B,
                     basis=basis, basis_hypers=basis_hypers,
                     observation=observation, observation_hypers=observation_hypers,
                     activation=activation, activation_hypers=activation_hypers,
                     bias=bias, bias_hypers=bias_hypers,
                     background=background, background_hypers=background_hypers,
                     weights=weights, weight_hypers=weight_hypers,
                     network=network, network_hypers=network_hypers,
                     n_iters_per_resample=n_iters_per_resample,
                     hdp_hmm_hypers=hdp_hmm_hypers)

class NegativeBinomialHDPHMM(_HDPHMMPopulationMixin, WeakLimitHDPHMM):
    _population_class = NegativeBinomialPopulationDistribution


class NegativeBinomialHDPHSMM(_HDPHSMMPopulationMixin, WeakLimitHDPHSMM):
    _population_class = NegativeBinomialPopulationDistribution


class NegativeBinomialEmptyHDPHMM(_HDPHMMPopulationMixin, WeakLimitHDPHMM):
    _population_class = NegativeBinomialEmptyPopulationDistribution


class NegativeBinomialEmptyHDPHSMM(_HDPHSMMPopulationMixin, WeakLimitHDPHSMM):
    _population_class = NegativeBinomialEmptyPopulationDistribution



    #
    # def __init__(self, N, M,
    #              dt=1.0, dt_max=10.0, B=5,                  # Population parameters
    #              basis=None, basis_hypers={},
    #              observation=None, observation_hypers={},
    #              activation=None, activation_hypers={},
    #              bias=None, bias_hypers={},
    #              background=None, background_hypers={},
    #              weights=None, weight_hypers={},
    #              network=None, network_hypers={},
    #              n_iters_per_resample=10,
    #              hdp_hmm_hypers={},                          # HDP-HMM hyperparameters
    #              duration_hypers={}
    #              ):
    #     """
    #     :param N:   The number of neurons
    #     :param M:   The maximum number of latent states
    #
    #     The remainder of the parameters are specified in the Population
    #     of HMM definitions.
    #     :return:
    #     """
    #     self.N = N
    #     self.M = M
    #
    #     # Initialize duration distributions
    #     self.duration_hypers = copy.deepcopy(self._default_duration_hypers)
    #     self.duration_hypers.update(duration_hypers)
    #     self.dur_distns = [self._duration_class(**self.duration_hypers) for m in range(self.M)]
    #
    #     hdp_hmm_hypers["dur_distns"] = self.dur_distns
    #
    #     super(NegativeBinomialHDPHSMM, self).\
    #         __init__(N, M, dt=dt, dt_max=dt_max, B=B,
    #                  basis=basis, basis_hypers=basis_hypers,
    #                  observation=observation, observation_hypers=observation_hypers,
    #                  activation=activation, activation_hypers=activation_hypers,
    #                  bias=bias, bias_hypers=bias_hypers,
    #                  background=background, background_hypers=background_hypers,
    #                  weights=weights, weight_hypers=weight_hypers,
    #                  network=network, network_hypers=network_hypers,
    #                  n_iters_per_resample=n_iters_per_resample,
    #                  hdp_hmm_hypers=hdp_hmm_hypers)