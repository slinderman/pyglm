"""
Activation models.
"""
import abc
import numpy as np

from pyglm.abstractions import Component
from pyglm.utils.profiling import line_profiled

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

    @property
    def bias_model(self):
        return self.population.bias_model

    @property
    def background_model(self):
        return self.population.background_model

    @property
    def weight_model(self):
        return self.population.weight_model

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

    @line_profiled
    def compute_residual(self, augmented_data, bias=None, synapse=None, bkgd=None):
        """
        Compute the residual activation for either the bias or the specified synapse.
        """
        N = self.N
        T = augmented_data["T"]
        F = augmented_data["F"]
        W = self.weight_model.W

        assert bias is not None or synapse is not None
        n_pre, n_post = self._get_n(bias, synapse)

        # compute psi, excluding the bias or synapse, whichever is specified
        psi = np.zeros(T)

        if bias is None:
            psi += self.bias_model.b[None, n_post]

        # Only compute residual if W is nonzero
        if not np.allclose(W[:,n_post], 0):
            for nn in xrange(N):
                if nn == n_pre:
                    continue
                psi += np.dot(F[:,nn,:], W[nn, n_post, :])

        if bkgd is None:
            psi += self.background_model.mean_background_activation(augmented_data)[:,n_post]

        return psi


class DeterministicActivation(_ActivationBase):
    """
    Deterministic activation. We just pass the activation through unchanged.
    """
    def compute_psi(self, augmented_data):
        N = self.N
        T = augmented_data["T"]
        F = augmented_data["F"]
        W = self.weight_model.W_effective

        # compute psi
        # psi = np.zeros((T,N))
        # psi += self.bias_model.b[None, :]
        #
        # for n_post in xrange(N):
        #     psi[:,n_post] += np.tensordot(F, W[:,n_post,:], axes=((1,2), (0,1)))

        psi = np.zeros((T,N))
        if not np.allclose(W, 0):
            np.einsum("tmb,mnb->tn", F, W, out=psi)
        psi += self.bias_model.b[None, :]

        # Add background activations
        psi += self.background_model.mean_background_activation(augmented_data)

        return psi

    def rvs(self, X):
        return X

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

    def new_precision(self, augmented_data):
        obs = self.observation_model
        return obs.omega(augmented_data)

    @line_profiled
    def mean_dot_precision(self, augmented_data, bias=None, synapse=None, psi_other=None):
        F = augmented_data["F"]
        obs = self.observation_model

        # If psi_other is not given, compute it.
        if psi_other is None:
            psi_other = self.compute_residual(augmented_data, bias, synapse)

        n_pre, n_post = self._get_n(bias, synapse)

        trm1 = obs.kappa(augmented_data)[:,n_post] \
               - psi_other * obs.omega(augmented_data)[:,n_post]

        if bias is not None:
            return trm1.sum()
        else:
            return trm1.dot(F[:,n_pre,:])

    def new_mean(self, augmented_data):
        obs = self.observation_model
        return obs.kappa(augmented_data) / obs.omega(augmented_data)

    ### Mean Field
    def meanfieldupdate(self, augmented_data):
        pass

    def mf_expected_activation(self, augmented_data):
        F = augmented_data["F"]
        Psi = np.zeros((augmented_data["T"], self.N))
        Psi += self.bias_model.mf_expected_bias()[None, :]

        W = self.weight_model.mf_expected_W()
        # for n_post in xrange(self.N):
        #     Psi[:,n_post] += np.tensordot(F, W[:,n_post,:], axes=((1,2), (0,1)))
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                Psi[:,n_post] += np.dot(F[:,n_pre,:], W[n_pre,n_post,:])

        return Psi

    def mf_marginal_variance_activation(self, augmented_data):
        F = augmented_data["F"]

        # Add the variance of the bias
        V_psi = np.zeros((augmented_data["T"], self.N))
        V_psi += self.bias_model.mf_sigma_b[None,:]

        # Add the variance of the weights
        # for n_post in xrange(self.N):
        #     Psi[:,n_post] += np.tensordot(F, W[:,n_post,:], axes=((1,2), (0,1)))
        for n_pre in xrange(self.N):
            F_pre = F[:,n_pre,:]
            for n_post in xrange(self.N):
                Sigma = self.weight_model.mf_Sigma[n_pre, n_post, :, :]

                # Get the marginal variance from F Sigma F.T
                # i.e., the diagonal of the TxT covariance matrix
                V_psi[:,n_post] += (F_pre.dot(Sigma) * F_pre).sum(1)

        return V_psi

    def mf_expected_residual(self, augmented_data, bias=None, synapse=None):
        """
        Compute the expected residual activation for either the bias or the specified synapse.
        """
        N = self.N
        T = augmented_data["T"]
        F = augmented_data["F"]
        W = self.weight_model.mf_expected_W()
        b = self.bias_model.mf_expected_bias()

        assert bias is not None or synapse is not None
        n_pre, n_post = self._get_n(bias, synapse)

        # compute psi, excluding the bias or synapse, whichever is specified
        psi = np.zeros(T)

        if bias is None:
            psi += b[None, n_post]

        for nn in xrange(N):
            if nn == n_pre:
                continue
            psi += np.dot(F[:,nn,:], W[nn, n_post, :])

        return psi

    def mf_sample_activation(self, augmented_data, N_samples=1):
        """
        Sample an activation
        :param Xs:
        :return:
        """
        psis = np.zeros((N_samples, augmented_data["T"], self.N))
        for smpl in xrange(N_samples):

            # Resample from the mean field distribution
            self.bias_model.resample_from_mf(augmented_data)
            self.weight_model.resample_from_mf(augmented_data)

            # Compute psi under this sample
            psis[smpl, :,:] = self.compute_psi(augmented_data)

        return psis

    def mf_sample_marginal_activation(self, augmented_data, N_samples=1):
        """
        Sample activations from their marginal distribution
        :param Xs:
        :return:
        """
        mu = self.mf_expected_activation(augmented_data)
        var = self.mf_marginal_variance_activation(augmented_data)
        std = np.sqrt(var)

        shp = (N_samples, augmented_data["T"], self.N)
        psis = mu[None, :,:] + std[None, :,:] * np.random.randn(*shp)
        return psis

    def mf_precision(self, augmented_data, bias=None, synapse=None):
        F = augmented_data["F"]
        obs = self.observation_model

        n_pre, n_post = self._get_n(bias, synapse)

        if bias is not None:
            return obs.mf_expected_omega(augmented_data)[:,n_post].sum()
        else:
            E_omega = obs.mf_expected_omega(augmented_data)[:,n_post]
            F_pre = F[:,n_pre,:]
            return (F_pre * E_omega[:,None]).T.dot(F_pre)

    def mf_mean_dot_precision(self, augmented_data,
                              bias=None, synapse=None,
                              E_psi_other=None):
        F = augmented_data["F"]
        obs = self.observation_model
        # residual = self.mf_expected_residual(augmented_data, bias, synapse)

        # If psi_other is not given, compute it.
        if E_psi_other is None:
            E_psi_other = self.mf_expected_residual(augmented_data, bias, synapse)

        n_pre, n_post = self._get_n(bias, synapse)

        trm1 = obs.kappa(augmented_data)[:,n_post] \
               - E_psi_other * obs.mf_expected_omega(augmented_data)[:,n_post]

        if bias is not None:
            return trm1.sum()
        else:
            return trm1.dot(F[:,n_pre,:])

    def get_vlb(self, augmented_data):
        return 0

    def resample_from_mf(self, augmented_data):
        pass

    ### SVI
    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        pass

class GaussianNoiseActivation(_ActivationBase):
    """
    The rate is the activation plus Gaussian noise.

    Each neuron gets its own noise scale.
    Each dataset gets an accompanying rate.
    """
    pass