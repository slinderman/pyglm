import numpy as np
from scipy.special import gammaln

from pyglm.deps.pybasicbayes.abstractions import GibbsSampling, ModelGibbsSampling
from pyglm.deps.pybasicbayes.distributions import GaussianFixedCov, GaussianFixed
from pyglm.deps.pybasicbayes.util.stats import sample_discrete_from_log
from pyglm.internals.distributions import InverseGamma
from pyglm.internals.observations import AugmentedNegativeBinomialCounts, AugmentedBernoulliCounts
from pyglm.synapses import GaussianVectorSynapse, SpikeAndSlabGaussianVectorSynapse


class _NeuronBase(GibbsSampling, ModelGibbsSampling):
    """
    Encapsulates the shared functionality of neurons with connections
    to the rest of the population. The observation model (e.g. Bernoulli,
    Poisson, Negative Binomial) need not be specified.
    """
    _synapse_class = GaussianVectorSynapse

    def __init__(self, n, population,
                 n_iters_per_resample=1,
                 alpha_0=3.0, beta_0=0.5):
        self.n = n
        self.population = population
        self.N = self.population.N
        self.n_iters_per_resample = n_iters_per_resample

        # Keep a list of spike train data
        self.data_list = []

        # Create the components of the neuron
        # self.noise_model = GaussianFixedMean(mu=np.zeros(1,),
        #                                  nu_0=1,
        #                                  lmbda_0=1*np.eye(1))
        self.noise_model = InverseGamma(alpha_0=alpha_0, beta_0=beta_0)

        # TODO: Remove this debugging value
        self.noise_model.sigma = 1.0
        # self.noise_model  = GaussianFixed(mu=np.zeros(1,), sigma=0.1 * np.eye(1))

        self.bias_model = GaussianFixedCov(mu_0=np.reshape(self.population.bias_prior.mu, (1,)),
                                      sigma_0=np.reshape(self.population.bias_prior.sigmasq, (1,1)),
                                      sigma=self.noise_model.sigma * np.eye(1))

        self.synapse_models = []
        for n_pre in range(self.N):
            self.synapse_models.append(
                self._synapse_class(self,
                                    n_pre,
                                    eta=self.eta * np.eye(1),
                                    ))


        # TODO: B and Ds are redundant. Switch to Ds since it is more flexible
        # and will eventually allow us to handle stimuli
        self.B = self.population.B
        self.Ds = np.array([syn.D_in for syn in self.synapse_models])

    @property
    def eta(self):
        return self.noise_model.sigma

    @eta.setter
    def eta(self, value):
        self.noise_model.sigma = value

        self.bias_model.setsigma(self.noise_model.sigma * np.eye(1))
        for syn in self.synapse_models:
            syn.eta = self.noise_model.sigma * np.eye(1)

    @property
    def bias(self):
        return self.bias_model.mu

    @bias.setter
    def bias(self, value):
        self.bias_model.mu = value

    @property
    def An(self):
        return np.array([syn.A for syn in self.synapse_models])

    @An.setter
    def An(self, value):
        for A,syn in zip(value, self.synapse_models):
            syn.A = A

    @property
    def weights(self):
        return np.array([syn.w for syn in self.synapse_models])

    @weights.setter
    def weights(self, value):
        for w,syn in zip(value, self.synapse_models):
            syn.set_weights(w)

    @property
    def parameters(self):
        return self.An, self.weights, self.eta, self.bias

    @parameters.setter
    def parameters(self, value):
        self.An, self.weights, self.eta, self.bias = value

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
        sigma = self.noise_model.sigma
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


class _GibbsNeuron(_NeuronBase):

    def resample(self,data=[]):
        for d in data:
            self.add_data(self._get_Xs(d), self._get_S(d)[:,self.n])

        for itr in xrange(self.n_iters_per_resample):
            self.resample_model()

        for _ in data[::-1]:
            self.data_list.pop()

        assert len(self.data_list) == 0

    def resample_model(self,
                       do_resample_psi=True,
                       do_resample_psi_from_prior=False,
                       do_resample_aux=True,
                       do_resample_bias=True,
                       do_resample_synapses=True,
                       do_resample_sigma=True):

        # import matplotlib.pyplot as plt
        for augmented_data in self.data_list:
            # Sample omega given the data and the psi's derived from A, sigma, and X
            augmented_data.resample(do_resample_psi=do_resample_psi,
                                    do_resample_psi_from_prior=do_resample_psi_from_prior,
                                    do_resample_aux=do_resample_aux)


        # d0 = self.data_list[0]
        # plt.plot(d0.psi, label='psi')
        # plt.plot(self.mean_activation(d0.X), label='act0')

        # Resample the bias model and the synapse models
        if do_resample_bias:
            self.resample_bias()

        # plt.plot(self.mean_activation(d0.X),'--r', label='act1')

        if do_resample_synapses:
            self.resample_synapse_models()

        # plt.plot(self.mean_activation(d0.X), '-m', label='act2')

        # Resample the noise variance sigma
        if do_resample_sigma:
            self.resample_sigma()

        # plt.legend()
        # plt.show()

    def resample_sigma(self):
        """
        Resample the noise variance phi.

        :return:
        """
        # import pdb; pdb.set_trace()
        residuals = []
        for data in self.data_list:
            residuals.append((data.psi - self.mean_activation(data.X))[:,None])

        self.noise_model.resample(residuals)

        # Update the synapse model covariances
        self.bias_model.setsigma(self.noise_model.sigma * np.eye(1))
        for syn in self.synapse_models:
            syn.eta = self.noise_model.sigma * np.eye(1)

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


class _MeanFieldNeuron(_NeuronBase):

    def __init__(self, n, population,
                 n_iters_per_resample=1,
                 alpha_0=3.0, beta_0=0.5):
        super(_MeanFieldNeuron, self).__init__(n, population, n_iters_per_resample, alpha_0, beta_0)

        # The GaussianFixedCov doesn't have a meanfield update yet so we'll implement it here
        self.mf_mu_bias = self.population.bias_prior.mu
        self.mf_sigma_bias = self.population.bias_prior.sigmasq

    @property
    def mf_rho(self):
        return np.array([syn.mf_rho for syn in self.synapse_models])

    @property
    def mf_mu_w(self):
        return np.array([syn.mf_mu_w for syn in self.synapse_models])

    @property
    def mf_Sigma_w(self):
        return np.array([syn.mf_Sigma_w for syn in self.synapse_models])

    def mf_mean_activation(self, Xs):
        T = Xs[0].shape[0]
        mu = np.zeros((T,))
        mu += self.mf_mu_bias
        for X,syn in zip(Xs, self.synapse_models):
            mu += syn.mf_predict(X)

        return mu

    def meanfield_coordinate_descent_step(self):
        for d in self.data_list:
            d.meanfield_update_psi()

        self.meanfield_update_bias()
        self.meanfield_update_synapses()

    def meanfield_update_synapses(self):
        """
        Jointly resample the spike and slab indicator variables and synapse models
        :return:
        """
        for n_pre in range(self.N):
            syn = self.synapse_models[n_pre]

            # Compute covariates and the predictions
            if len(self.data_list) > 0:
                X_pres = []
                residuals = []
                for d in self.data_list:
                    X_pres.append(d.X[n_pre])

                    mu_other = self.mf_mu_bias * np.ones_like(d.psi)
                    for n_other,X,syn_other in zip(np.arange(self.N), d.X, self.synapse_models):
                        if n_other != n_pre:
                            mu_other += syn_other.mf_predict(X)

                    # Use mean field activation to compute residuals
                    residual = (d.mf_mu_psi - mu_other)[:,None]
                    residuals.append(residual)

                X_pres = np.vstack(X_pres)
                residuals = np.vstack(residuals)
                X_and_residuals = np.hstack((X_pres,residuals))

                # Call the synapse's mean field update
                syn.meanfieldupdate(X_and_residuals, None)

    def meanfield_update_bias(self):
        """
        Update the variational parameters for the bias
        """
        if len(self.data_list) > 0:
            residuals = []
            for d in self.data_list:
                mu = np.zeros_like(d.psi)
                for X,syn in zip(d.X, self.synapse_models):
                    mu += syn.mf_predict(X)

                # Use mean field activation to compute residuals
                residual = (d.mf_mu_psi - mu)[:,None]
                residuals.append(residual)
            residuals = np.vstack(residuals)

            # TODO: USE MF ETA to compute residual
            T = residuals.shape[0]
            self.mf_sigma_bias = 1.0/(T/self.eta + 1.0/self.bias_model.sigma_0)
            self.mf_mu_bias = self.mf_sigma_bias * (residuals.sum()/self.eta +
                                                    self.bias_model.mu_0/self.bias_model.sigma_0)

            self.mf_sigma_bias = np.asscalar(self.mf_sigma_bias)
            self.mf_mu_bias = np.asscalar(self.mf_mu_bias)

        else:
            self.mf_sigma_bias = self.bias_model.sigma_0
            self.mf_mu_bias = self.bias_model.mu_0



class _SpikeAndSlabNeuron(_NeuronBase):
    """
    Encapsulates the shared functionality of neurons with sparse connectivity
    to the rest of the population. The observation model (e.g. Bernoulli,
    Poisson, Negative Binomial) need not be specified.
    """

    _synapse_class = SpikeAndSlabGaussianVectorSynapse


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

class BernoulliNeuron(_GibbsNeuron, _MeanFieldNeuron, _AugmentedDataMixin):

    def __init__(self, n, population, n_iters_per_resample=1,
                 alpha_0=3.0, beta_0=0.5):
        super(BernoulliNeuron, self).\
            __init__(n, population,
                     n_iters_per_resample=n_iters_per_resample,
                     alpha_0=alpha_0, beta_0=beta_0)

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


class BernoulliSpikeAndSlabNeuron(BernoulliNeuron, _SpikeAndSlabNeuron):
    pass


class NegativeBinomialNeuron(_GibbsNeuron, _MeanFieldNeuron, _AugmentedDataMixin):
    def __init__(self, n, population, xi=10,
                 n_iters_per_resample=1,
                 alpha_0=3.0, beta_0=0.5):
        super(NegativeBinomialNeuron, self).\
            __init__(n, population, n_iters_per_resample=n_iters_per_resample,
                     alpha_0=alpha_0, beta_0=beta_0)

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


class NegativeBinomialSpikeAndSlabNeuron(NegativeBinomialNeuron, _SpikeAndSlabNeuron):
    pass