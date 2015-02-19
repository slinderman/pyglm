"""
Create switching population models by using the population model
as an observation distribution in a hidden Markov model.
"""
from pyhsmm.models import WeakLimitHDPHMM
from pyglm.distributions import PopulationDistribution

class SwitchingPopulation(WeakLimitHDPHMM):
    """
    A switching population model with Markovian dynamics.
    """
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
        # Create observation models
        self.populations = \
            [PopulationDistribution(N,
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

        # Initialize the switching model with the populations
        super(SwitchingPopulation, self).\
            __init__(obs_distns=self.populations, **hdp_hmm_hypers)

    def add_data(self, data, stateseq=None, **kwargs):
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
        packed_data = self.populations[0].pack_spike_train(data)

        super(SwitchingPopulation, self).add_data(packed_data, stateseq=stateseq, **kwargs)

    def compute_rate(self):
        """
        Compute the "firing rate" of the model with its current state sequence.
        :return:
        """
        raise NotImplementedError()

    @property
    def hidden_state_sequence(self):
        hidden_states = [s.stateseq for s in self.states_list]
        return hidden_states
