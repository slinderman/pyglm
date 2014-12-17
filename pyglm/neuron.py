import numpy as np
from scipy.special import gammaln

from deps.pybasicbayes.abstractions import GibbsSampling, ModelGibbsSampling
from deps.pybasicbayes.distributions import GaussianFixedMean, GaussianFixedCov, GaussianFixed
from deps.pybasicbayes.util.stats import sample_discrete_from_log
from pyglm.internals.observations import AugmentedNegativeBinomialCounts, AugmentedBernoulliCounts
from pyglm.synapses import GaussianVectorSynapse


class _NeuronBase(GibbsSampling, ModelGibbsSampling):
    """
    Encapsulates the shared functionality of neurons with connections
    to the rest of the population. The observation model (e.g. Bernoulli,
    Poisson, Negative Binomial) need not be specified.
    """
    def __init__(self, n, population,
                 n_iters_per_resample=1):
        self.n = n
        self.population = population
        self.N = self.population.N
        self.n_iters_per_resample = n_iters_per_resample

        # Keep a list of spike train data
        self.data_list = []

        # Create the components of the neuron
        # self.noise_model = GaussianFixedMean(mu=np.zeros(1,),
        #                                  nu_0=10,
        #                                  lmbda_0=10*np.eye(1))
        self.noise_model  = GaussianFixed(mu=np.zeros(1,), sigma=1.0 * np.eye(1))

        self.bias_model = GaussianFixedCov(mu_0=np.reshape(self.population.bias_prior.mu, (1,)),
                                      lmbda_0=np.reshape(self.population.bias_prior.sigmasq, (1,1)),
                                      sigma=self.noise_model.sigma)

        self.synapse_models = []
        for n_pre in range(self.N):
            self.synapse_models.append(
                GaussianVectorSynapse(self,
                                      n_pre,
                                      sigma=self.noise_model.sigma,
                                      ))


        # TODO: B and Ds are redundant. Switch to Ds since it is more flexible
        # and will eventually allow us to handle stimuli
        self.B = self.population.B
        self.Ds = np.array([syn.D_in for syn in self.synapse_models])

    @property
    def sigma(self):
        return self.noise_model.sigma

    @sigma.setter
    def sigma(self, value):
        self.noise_model.setsigma(value)

        self.bias_model.setsigma(self.noise_model.sigma)
        for syn in self.synapse_models:
            syn.sigma = self.noise_model.sigma

    @property
    def bias(self):
        return self.bias_model.mu

    @bias.setter
    def bias(self, value):
        self.bias_model.mu = value

    @property
    def An(self):
        return np.ones(self.N)

    @An.setter
    def An(self, value):
        pass

    @property
    def weights(self):
        return np.array([syn.w for syn in self.synapse_models])

    @weights.setter
    def weights(self, value):
        for w,syn in zip(value, self.synapse_models):
            syn.set_weights(w)

    @property
    def parameters(self):
        return self.weights, self.sigma, self.bias

    @parameters.setter
    def parameters(self, value):
        self.weights, self.sigma, self.bias = value

    ## These must be implemented by base classes
    def mean(self, Xs):
        raise NotImplementedError()

    def std(self, Xs):
        raise NotImplementedError()

    def _internal_log_likelihood(self, Xs, y):
        raise NotImplementedError()

    def sample_observations(self, psi):
        raise NotImplementedError("Base classes must override this!")

    def mean_activation(self, Xs):
        T = Xs[0].shape[0]
        mu = np.zeros((T,))
        mu += self.bias
        for X,syn in zip(Xs, self.synapse_models):
            mu += syn.predict(X)

        return mu

    def _get_Xs(self,d):
        B, N = self.B, self.N
        return [d[:,idx:idx+B] for idx in range(0,N*B,B)]

    def _get_S(self,d):
        return d[:,-self.N:]

    def pop_data(self):
        return self.data_list.pop()

    def log_likelihood(self, x):
        return self._internal_log_likelihood(
                self._get_Xs(x), self._get_S(x)[:,self.n])

    def heldout_log_likelihood(self,x):
        return self._internal_log_likelihood(self._get_Xs(x),
                                             self._get_S(x)[:,self.n]
                                            ).sum()


    def rvs(self, X=None, Xs=None, size=1, return_xy=False):
        sigma = np.asscalar(self.noise_model.sigma)
        if X is None and Xs is None:
            T = size
            Xs = []
            for D in self.Ds:
                Xs.append(np.random.normal(size=(T,D)))
        else:
            if X is not None and Xs is None:
                Xs = self._get_Xs(X)

            T = Xs[0].shape[0]
            Ts = np.array([X.shape[0] for X in Xs])
            assert np.all(Ts == T)

        psi = self.mean_activation(Xs) + np.sqrt(sigma) * np.random.normal(size=(T,))

        # Sample the negative binomial. Note that the definition of p is
        # backward in the Numpy implementation
        y = self.sample_observations(psi)

        return (Xs,y) if return_xy else y

    def generate(self,keep=True,**kwargs):
        return self.rvs(), None

    def resample(self,data=[]):
        for d in data:
            self.add_data(self._get_Xs(d), self._get_S(d)[:,self.n])

        for itr in xrange(self.n_iters_per_resample):
            self.resample_model()

        for _ in data[::-1]:
            self.data_list.pop()

        assert len(self.data_list) == 0

    def resample_model(self,
                       do_resample_data=True,
                       do_resample_bias=True,
                       do_resample_synapses=True,
                       do_resample_sigma=True,
                       do_resample_counts=False):

        # TODO: Cache the X \dot w calculations
        if do_resample_data:
            for augmented_data in self.data_list:
                # Sample omega given the data and the psi's derived from A, sigma, and X
                augmented_data.resample()

        # Resample the bias model and the synapse models
        if do_resample_bias:
            self.resample_bias()

        if do_resample_synapses:
            self.resample_synapse_models()

        # Resample the noise variance sigma
        if do_resample_sigma:
            self.resample_sigma()

    def resample_sigma(self):
        """
        Resample the noise variance phi.

        :return:
        """
        residuals = []
        for data in self.data_list:
            residuals.append((data.psi - self.mean_activation(data.X))[:,None])

        self.noise_model.resample(residuals)

        # Update the synapse model covariances
        self.bias_model.setsigma(self.noise_model.sigma)
        for syn in self.synapse_models:
            syn.sigma = self.noise_model.sigma

    def resample_bias(self):
        """
        Resample the bias given the weights and psi
        :return:
        """
        residuals = []
        for data in self.data_list:
            residuals.append(data.psi - (self.mean_activation(data.X) - self.bias_model.mu))

        if len(residuals) > 0:
            residuals = np.concatenate(residuals)

            # Residuals must be a Nx1 vector
            self.bias_model.resample(residuals[:,None])

        else:
            self.bias_model.resample([])

    def resample_synapse_models(self):
        """
        Jointly resample the spike and slab indicator variables and synapse models
        :return:
        """
        for n_pre in range(self.N):
            syn = self.synapse_models[n_pre]

            # Compute covariates and the predictions
            if len(self.data_list) > 0:
                Xs = []
                residuals = []
                for d in self.data_list:
                    Xs.append(d.X[n_pre])
                    residual = (d.psi - (self.mean_activation(d.X) - syn.predict(d.X[n_pre])))[:,None]
                    residuals.append(residual)

                Xs = np.vstack(Xs)
                residuals = np.vstack(residuals)

                X_and_residuals = np.hstack((Xs,residuals))
                syn.resample(X_and_residuals)


class _SparseNeuronMixin(_NeuronBase):
    """
    Encapsulates the shared functionality of neurons with sparse connectivity
    to the rest of the population. The observation model (e.g. Bernoulli,
    Poisson, Negative Binomial) need not be specified.
    """


    def __init__(self, n, population,
                 n_iters_per_resample=1,
                 rho_s=None,
                 An=None):

        super(_SparseNeuronMixin, self).__init__(n, population, n_iters_per_resample)

        if rho_s is not None:
            self.rho_s = rho_s
        elif population is not None:
            self.rho_s = self.population.network.rho[:,self.n]
        else:
            raise Exception("Rho must be specified or population must be given")

        self._An = An

        # For each parameter, make sure it is either specified or given a prior
        if An is None:
            self.An = np.ones(self.N)
            self.resample_synapse_models()

    @property
    def An(self):
        return self._An

    @An.setter
    def An(self, value):
        self._An = value

    @property
    def parameters(self):
        return self.An, self.weights, self.sigma, self.bias

    @parameters.setter
    def parameters(self, value):
        self.An, self.weights, self.sigma, self.bias = value

    def mean_activation(self, Xs):
        T = Xs[0].shape[0]
        mu = np.zeros((T,))
        mu += self.bias
        for X,A,syn in zip(Xs, self.An, self.synapse_models):
            if A > 0:
                mu += syn.predict(X)

        return mu

    ### Gibbs sampling
    def resample_synapse_models(self):
        """
        Jointly resample the spike and slab indicator variables and synapse models
        :return:
        """
        for n_pre in range(self.N):
            rho = self.rho_s[n_pre]
            synapse = self.synapse_models[n_pre]

            # Compute residual
            self.An[n_pre] = 0  # Make sure mu is computed without the current regression model
            if len(self.data_list) > 0:
                # residuals = np.vstack([(d.psi - self.mean_activation(d.X))[:,None] for d in self.data_list])
                residuals = np.vstack([(d.psi - (self.mean_activation(d.X) - synapse.predict(d.X[n_pre])))[:,None]
                                       for d in self.data_list])
                Xs = np.vstack([d.X[n_pre] for d in self.data_list])
                X_and_residuals = np.hstack((Xs,residuals))

                # Compute log Pr(A=0|...) and log Pr(A=1|...)
                if rho > 0.:
                    lp_A = np.zeros(2)
                    lp_A[0] = np.log(1.0-rho) + GaussianFixed(np.array([0]), self.noise_model.sigma)\
                                                    .log_likelihood(residuals).sum()
                    lp_A[1] = np.log(rho) + synapse.log_marginal_likelihood(X_and_residuals).sum()
                else:
                    lp_A = np.log([1.,0.])

            else:
                # Compute log Pr(A=0|...) and log Pr(A=1|...)
                lp_A = np.zeros(2)
                lp_A[0] = np.log(1.0-rho)
                lp_A[1] = np.log(rho)

                X_and_residuals = []


            # Sample the spike variable
            # self.As[m] = log_sum_exp_sample(lp_A)
            self.An[n_pre] = sample_discrete_from_log(lp_A)

            # Sample the slab variable
            if self.An[n_pre]:
                synapse.resample(X_and_residuals)


class _AugmentedDataMixin:
    def _augment_data(self, observation_class, Xs, counts):
        """
        For some observation models we have auxiliary variables that
        need to persist from one MCMC iteration to the next. To implement this, we
        keep the data around as a class variable. This is pretty much the same as
        what is done with the state labels in PyHSMM.

        :param data:
        :return:
        """

        assert counts.ndim == 1 and np.all(counts >= 0)
        counts = counts.astype(np.int)

        # Return an augmented counts object
        return observation_class(Xs, counts, self)

class BernoulliNeuron(_NeuronBase, _AugmentedDataMixin):

    def __init__(self, n, population, n_iters_per_resample=1, ):
        super(BernoulliNeuron, self).\
            __init__(n, population,
                     n_iters_per_resample=n_iters_per_resample)

    def add_data(self, data=[]):
        if isinstance(data, np.ndarray):
            data = [data]
        for d in data:
            Xs = self._get_Xs(d)
            counts = self._get_S(d)[:,self.n]
            self.data_list.append(self._augment_data(AugmentedBernoulliCounts, Xs, counts))

    def sample_observations(self, psi):
        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))
        return np.random.rand(*p.shape) < p

    def mean(self, Xs):
        """
        Compute the mean number of spikes for a given set of regressors, Xs

        :param Xs:
        :return:
        """
        return 1./(1+np.exp(-self.mean_activation(Xs)))

    def std(self, Xs):
        p = self.mean(Xs)
        return np.sqrt(p*(1-p))

    def _internal_log_likelihood(self, Xs, y):
        psi = self.mean_activation(Xs)

        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))
        p = np.clip(p, 1e-16, 1-1e-16)

        ll = (y * np.log(p) + (1-y) * np.log(1-p))
        return ll


class BernoulliSparseNeuron(BernoulliNeuron, _SparseNeuronMixin, _AugmentedDataMixin):
    pass


class NegativeBinomialNeuron(_NeuronBase, _AugmentedDataMixin):
    def __init__(self, n, population, xi=10,
                 n_iters_per_resample=1):
        super(NegativeBinomialNeuron, self).\
            __init__(n, population, n_iters_per_resample=n_iters_per_resample)

        self.xi = xi

    def add_data(self, data=[]):
        if isinstance(data, np.ndarray):
            data = [data]

        for d in data:
            Xs = self._get_Xs(d)
            counts = self._get_S(d)[:,self.n]
            self.data_list.append(self._augment_data(AugmentedNegativeBinomialCounts, Xs, counts))

    def sample_observations(self, psi):
        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))
        return np.random.negative_binomial(self.xi, 1-p)

    def mean(self, Xs):
        """
        Compute the mean number of spikes for a given set of regressors, Xs

        :param Xs:
        :return:
        """
        return self.xi * np.exp(self.mean_activation(Xs))

    def std(self, Xs):
        lmbda = np.exp(self.mean_activation(Xs))
        return np.sqrt(self.xi * lmbda * (1+lmbda))

    def _internal_log_likelihood(self, Xs, y):
        psi = np.clip(self.mean_activation(Xs),-np.inf,100.)

        # Convert the psi's into the negative binomial rate parameter, p
        p = np.exp(psi) / (1.0 + np.exp(psi))
        p = np.clip(p, 1e-16, 1-1e-16)

        ll = gammaln(self.xi + y) - gammaln(self.xi) - gammaln(y+1) + \
             self.xi * np.log(1.0-p) + (y*np.log(p))

        return ll


class NegativeBinomialSparseNeuron(NegativeBinomialNeuron, _SparseNeuronMixin, _AugmentedDataMixin):
    pass