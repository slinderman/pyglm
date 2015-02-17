"""
Activation models.
"""
import abc
import numpy as np

from pyglmdos.abstractions import Component

class _ActivationBase(Component):
    """
    Base class for activations.
    """
    __metaclass__ =  abc.ABCMeta

    def __init__(self, population):
        self.population = population

    @property
    def N(self):
        return self.population.N

    @property
    def observation_model(self):
        return self.population.observation_model

    @abc.abstractmethod
    def compute_psi(self, augmented_data):
        raise NotImplementedError()

    def _get_n(self, bias=None, synapse=None):
        n_post = n_pre = None
        if bias is not None:
            n_post = bias
        else:
            n_pre, n_post = synapse
        return n_pre, n_post

    def compute_residual(self, augmented_data, bias=None, synapse=None):
        """
        Compute the residual activation for either the bias or the specified synapse.
        """
        N = self.N
        T = augmented_data["T"]
        F = augmented_data["F"]
        W = self.population.weight_model.W

        assert bias is not None or synapse is not None
        n_pre, n_post = self._get_n(bias, synapse)

        # compute psi, excluding the bias or synapse, whichever is specified
        psi = np.zeros(T)

        if bias is None:
            psi += self.population.bias_model.b[None, n_post]

        for nn in xrange(N):
            if nn == n_pre:
                continue
            psi += np.dot(F[:,nn,:], W[nn, n_post, :])

        return psi


class DeterministicActivation(_ActivationBase):
    """
    Deterministic activation. We just pass the activation through unchanged.
    """
    def compute_psi(self, augmented_data):
        N = self.N
        T = augmented_data["T"]
        F = augmented_data["F"]

        # compute psi
        psi = np.zeros((T,N))
        psi += self.population.bias_model.b[None, :]

        W = self.population.weight_model.W
        for n_post in xrange(N):
            psi[:,n_post] += np.tensordot(F, W[:,n_post,:], axes=((1,2), (0,1)))

        return psi

    def resample(self, augmented_data):
        pass

    def precision(self, augmented_data, bias=None, synapse=None):
        F = augmented_data["F"]
        obs = self.observation_model

        n_pre, n_post = self._get_n(bias, synapse)

        if bias is not None:
            return obs.omega(augmented_data)[:,n_post].sum()
        else:
            omega = obs.omega(augmented_data)[:,n_post]
            F_pre = F[:,n_pre,:]
            return (F_pre * omega[:,None]).T.dot(F_pre)

    def mean_dot_precision(self, augmented_data, bias=None, synapse=None):
        F = augmented_data["F"]
        obs = self.observation_model
        residual = self.compute_residual(augmented_data, bias, synapse)

        n_pre, n_post = self._get_n(bias, synapse)

        trm1 = obs.kappa(augmented_data)[:,n_post] - residual * obs.omega(augmented_data)[:,n_post]

        if bias is not None:
            return trm1.sum()
        else:
            return trm1.dot(F[:,n_pre,:])

    def rvs(self, X):
        return X


class GaussianNoiseActivation(_ActivationBase):
    """
    The rate is the activation plus Gaussian noise.

    Each neuron gets its own noise scale.
    Each dataset gets an accompanying rate.
    """
    pass