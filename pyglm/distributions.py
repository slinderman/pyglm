"""
Wrap the population models in a PyBasicBayes distribution.
This allows us to use the population as an observation model
for an HMM.
"""
import numpy as np

from pyglm.models import Population
from pyglm.deps.pybasicbayes.abstractions import GibbsSampling

class PopulationDistribution(Population, GibbsSampling):
    """
    A wrapper for a Population model that exposes an interface for
    inference in an enclosing HMM. The states of the HMM correspond
    to different population parameters, e.g. different background rates
    or different networks.
    """

    def __init__(self, N, n_iters_per_resample=10, **kwargs):
        super(PopulationDistribution, self).__init__(N, **kwargs)

        # Set the number of iterations per resample
        self.n_iters_per_resample = n_iters_per_resample

    def pack_spike_train(self, S):
        """
        Augment the spike train with its filtered counterpart.
        Then concatenate the filtered spike train and the spike train matrix.
        :param S: A TxN matrix of spike counts
        :return:
        """
        augmented_data = self.augment_data(S)
        T = augmented_data["T"]
        F = augmented_data["F"]

        # Flatten F
        F_flat = F.reshape((T, self.N*self.B))

        # Pack F and S together
        packed_data = np.hstack((F_flat, S))
        return packed_data

    def _unpack_data(self, packed_data):
        T = packed_data.shape[0]
        assert packed_data.shape[1] == self.N * self.B + self.N

        # Extract F
        F_flat = packed_data[:, :(self.N * self.B)]
        F      = F_flat.reshape((T, self.N, self.B))
        # Extract S
        S      = packed_data[:, (self.N * self.B):]
        S      = S.astype(np.int32)


        return S, F

    def rvs(self,size=[]):
        S = self.generate(keep=False, T=size)
        return S

    def log_likelihood(self, packed_data):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        # Make sure packed_data is a list
        assert len(self.data_list) == 0
        # Unpack the data and add it to the model data list
        S,F = self._unpack_data(packed_data)
        self.add_data(S, F=F)

        # Compute the log likelihood
        lls = super(PopulationDistribution, self).log_likelihood(self.data_list[-1]).sum(axis=1)

        # Remove the data from the datalist
        self.data_list.pop()

        assert len(self.data_list) == 0
        return lls

    def resample(self, data=[]):
        """
        The HMM calls resample and gives us a bunch of data points to use.
        The "data" consists of both the spike counts and the regressors.
        In our case, the regressors are the filtered spike train.

        :param data:
        :return:
        """
        assert len(self.data_list) == 0
        # Make sure packed_data is a list
        if not isinstance(data, list):
            data = [data]

        # Unpack the data and add it to the model data list
        for pdata in data:
            if pdata.size > 0:
                S,F = self._unpack_data(pdata)
                self.add_data(S, F=F)

        # Resample with the given data
        for itr in xrange(self.n_iters_per_resample):
            self.resample_model()

        # Remove the data from the datalist
        for pdata in data:
            if pdata.size > 0:
                self.data_list.pop()

        assert len(self.data_list) == 0