"""
Network models expose a probability of connection and a scale of the weights
"""
import abc

import numpy as np
from scipy.special import gammaln, psi

from pyglm.abstractions import Component

from pyglm.deps.pybasicbayes.util.stats import sample_discrete_from_log
from pyglm.deps.pybasicbayes.util.stats import sample_niw

from pyglm.internals.distributions import Bernoulli

# Import graph models from graphistician
from pyglm.deps.graphistician.networks import GaussianWeightedEigenmodel

class _NetworkComponent(Component):
    """
    Expose a graphistician network through the Component interface
    """
    __metaclass__ = abc.ABCMeta

    # Network class must be an instance of
    # GaussianWeightedNetworkDistribution
    _network_class = None

    def __init__(self, population, **network_hypers):
        self.population = population
        self.N = population.N
        self.B = population.B

        self._model = self._network_class(self.N, self.B, **network_hypers)

    @property
    def weight_model(self):
        return self.population.weight_model

    @property
    def P(self): return self._model.P

    @property
    def Mu(self): return self._model.Mu

    @property
    def Sigma(self): return self._model.Sigma

    def log_prior(self): return self._model.log_prior()

    # Gibbs sampling
    def resample(self, augmented_data):
        self._model.resample(self.weight_model)

    # Mean field
    def meanfieldupdate(self, augmented_data):
        self._model.meanfieldupdate(self.weight_model)

    def get_vlb(self, augmented_data):
        return self._model.get_vlb()

    def resample_from_mf(self, augmented_data):
        self._model.resample_from_mf()

    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        raise NotImplementedError()

class GaussianEigenmodel(_NetworkComponent):
    _network_class = GaussianWeightedEigenmodel
