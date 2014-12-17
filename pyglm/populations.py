"""
Top level model for a population of spike trains
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from deps.pybasicbayes.abstractions import GibbsSampling, ModelGibbsSampling
from deps.pybasicbayes.distributions import ScalarGaussianNIX, Gaussian
from latent import LatentClass, _LatentVariableBase
from pyglm.neuron import NegativeBinomialSparseNeuron, BernoulliSparseNeuron, BernoulliNeuron, NegativeBinomialNeuron
from pyglm.networks import ErdosRenyiNetwork, StochasticBlockNetwork, CompleteNetwork

class _PopulationOfNeuronsBase(GibbsSampling, ModelGibbsSampling):
    """
    Base class for a population spike train
    """
    def __init__(self, N,
                 basis,
                 latent_variable_class=None,
                 latent_variable_hypers={},
                 neuron_class=None,
                 neuron_hypers={},
                 network_class=None,
                 network_hypers={},
                 global_bias_class=ScalarGaussianNIX,
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0},
                 n_iters_per_resample=1
                 ):
        self.N = N

        # Instantiate a basis for filtering the spike trains
        self.basis = basis
        self.B = self.basis.B

        # Instantiate a model of latent variables, for example, types of neurons
        if latent_variable_class is not None:
            self.latent = latent_variable_class(self, **latent_variable_hypers)
        else:
            self.latent = _LatentVariableBase(self)

        # TODO: Also model stimuli

        # Instantiate the network object
        # The network needs a pointer to the population so that it
        # can get the number of neurons and the information about
        # the basis that is used to filter the spike trains
        self.network = network_class(self, **network_hypers)

        # Create global priors over parameters of the spike trains.
        # 1. Global prior over the mean and variance of the biases
        self.bias_prior = global_bias_class(**global_bias_hypers)

        # Instantiate a set of observations, one for each neuron
        # The spike trains take a  pointer back to the population
        # so that they can get the global priors.
        self.neuron_models = [neuron_class(n, self, **neuron_hypers)
                              for n in range(N)]

        self.n_iters_per_resample = n_iters_per_resample
    @property
    def biases(self):
        return np.array([neuron.bias for neuron in self.neuron_models])

    @biases.setter
    def biases(self, value):
        for n,neuron in enumerate(self.neuron_models):
            neuron.bias = value[n]

    @property
    def sigmas(self):
        return np.array([np.asscalar(neuron.sigma) for neuron in self.neuron_models])

    @sigmas.setter
    def sigmas(self,val):
        for neuron, sigma in zip(self.neuron_models,val):
            neuron.sigma = sigma

    @property
    def A(self):
        return np.array([neuron.An for neuron in self.neuron_models]).T

    @A.setter
    def A(self, value):
        for n,neuron in enumerate(self.neuron_models):
            neuron.An = value[:,n]

    @property
    def weights(self):
        return np.array([neuron.weights for neuron in self.neuron_models]).transpose([1,0,2])

    @weights.setter
    def weights(self, value):
        for n,neuron in enumerate(self.neuron_models):
            neuron.weights = value[:,n,:]

    @property
    def W_effective(self):
        A = self.A
        W = self.weights.sum(axis=-1)
        return A * W

    @property
    def bias_prior_prms(self):
        return self.bias_prior.mu, self.bias_prior.sigmasq

    @bias_prior_prms.setter
    def bias_prior_prms(self,val):
        self.bias_prior.mu, self.bias_prior.sigmasq = val

    @property
    def parameters(self):
        return (self.A, self.weights, self.biases, self.sigmas, self.bias_prior_prms) + \
                self.network.parameters # + self.latent.parameters

    @parameters.setter
    def parameters(self,val):
        self.A, self.weights, self.biases, self.sigmas, self.bias_prior_prms = val[:5]
        self.network.parameters = val[5:]

    @property
    def parameter_names(self):
        return ("A", "w", "bias", "sigma", "bias_prior") + \
                self.network.parameter_names # + self.latent.parameter_names


    def filter_spike_train(self, spiketrain):
        # Filter the spike train
        return self.basis.convolve_with_basis(spiketrain)


    def log_likelihood(self, data):
        assert isinstance(data,np.ndarray)
        return sum(neuron.log_likelihood(data) for neuron in self.neuron_models)

    def heldout_log_likelihood(self, data=[]):
        """
        Compute the log likelihood of held out data x

        :param data: Tx(NxB+N) array. The first NxB columns contain the
                     features of the spike train that we regress agains, and
                     the last N columns contain the spike counts.
        :return:
        """
        return sum(neuron.heldout_log_likelihood(data) for neuron in self.neuron_models)

    def add_data(self, data=[]):
        if isinstance(data, np.ndarray):
            data = [data]

        for neuron in self.neuron_models:
            neuron.add_data(data)

    def pop_data(self):
        datas = []
        for neuron in self.neuron_models:
            datas.append(neuron.pop_data())

        return datas

    def initialize_to_empty(self):
        for n,neuron in enumerate(self.neuron_models):
            M = np.sum([d.counts for d in neuron.data_list])
            T = np.sum([d.T  for d in neuron.data_list])
            if T > 0:
                p = np.float(M) / T
                # neuron.bias = np.array([logit(p)])
                neuron.weights *= 0

                if hasattr(neuron, 'An'):
                    neuron.An *= 0

    def initialize_with_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        print "Initializing with logistic regresion"
        for n_post,neuron in enumerate(self.neuron_models):
            sys.stdout.write('.')
            sys.stdout.flush()
            X = np.vstack([np.hstack(d.X) for d in neuron.data_list])
            counts = np.concatenate([d.counts for d in neuron.data_list])

            lr = LogisticRegression(fit_intercept=True)
            lr.fit(X,counts)

            w = lr.coef_
            bias = lr.intercept_

            neuron.weights = w.reshape((self.N,-1))
            neuron.An = np.ones_like(neuron.An)

            # Threshold the weights.
            # TODO: Get cutoff from network model
            rho = 0.5
            W = abs(neuron.weights.sum(1))
            Wthr = np.sort(W)[::-1][np.ceil(rho*self.N)]
            neuron.An = (abs(W) > Wthr) * np.ones_like(neuron.An)

            neuron.bias = bias
        print ""

    def resample(self,data=[]):
        if isinstance(data, np.ndarray):
            data = [data]

        for d in data:
            for neuron in self.neuron_models:
                neuron.add_data(d)

        for itr in xrange(self.n_iters_per_resample):
            self.resample_model()

        for _ in data[::-1]:
            for neuron in self.neuron_models:
                neuron.data_list.pop()

                assert len(neuron.data_list) == 0

    def resample_model(self,
                       do_resample_latent=True,
                       do_resample_network=True,
                       do_resample_bias_prior=True,
                       do_resample_bias=True,
                       do_resample_sigma=True,
                       do_resample_synapses=True,
                       do_resample_data=True,
                       do_resample_counts=False):
        """
        Resample the parameter of the model.

        :param data: Tx(NxB+N) array. The first NxB columns contain the
                     features of the spike train that we regress agains, and
                     the last N columns contain the spike counts.
        :return:
        """

        # Resample the spike train models
        for n,neuron in enumerate(self.neuron_models):
            neuron.resample_model(do_resample_bias=do_resample_bias,
                                  do_resample_sigma=do_resample_sigma,
                                  do_resample_synapses=do_resample_synapses,
                                  do_resample_data=do_resample_data,
                                  do_resample_counts=do_resample_counts)

        # Resample the network parameters with the given weights
        if do_resample_network:
            self.resample_network()

        # Resample latent variables of the population
        if do_resample_latent:
            self.resample_latent()

        # Resample global priors
        if do_resample_bias_prior:
            self.resample_bias_prior()
        # sys.stdout.write('\n')
        # sys.stdout.flush()

    def resample_latent(self):
        if self.latent is not None:
            self.latent.resample()

    def resample_network(self):
        self.network.resample(data=(self.A, self.weights))

    def resample_bias_prior(self):
        """
        Resample the global prior on biases
        :param data:
        :return:
        """
        self.bias_prior.resample(data=self.biases)
        for neuron in self.neuron_models:
            neuron.bias_model.mu_0=np.reshape(self.bias_prior.mu, (1,))
            neuron.bias_model.lmbda_0=np.reshape(self.bias_prior.sigmasq, (1,1)),

    def generate(self, keep=True, size=100, X_bkgd=None, verbose=False):
        """
        Simulate a spike train.

        :param size:    Number of time bins to simulate
        :param X_bkgd:  Optional. Background features that may be present.
                        This is useful if we want to simulate with an HMM
                        where some spikes may already have occurred.
        :return:
        """
        N = self.N
        assert isinstance(size, int), "Size must be an integer number of time bins"
        T = size
        B = self.basis.B    # Number of features per spike train
        L = self.basis.L    # Length of the impulse responses

        # Initialize output matrix of spike counts
        S = np.zeros((T,N))
        # Initialize the autoregressive feature matrix
        Xs = [np.zeros((T+L, B)) for n in range(N)]

        # If we have some features to start with, add them now
        if X_bkgd is not None:
            T_bkgd = min(T+L, X_bkgd.shape[0])
            for n in range(N):
                assert X_bkgd[n].shape[1] == B
                Xs[n][:T_bkgd,:] += X_bkgd[n]

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

            # Sample from the observation model for each spike train
            for n,neuron in enumerate(self.neuron_models):
                S[t,n] = self.neuron_models[n].rvs(size=1,
                                                   Xs=[X[t:t+1,:] for X in Xs])

                # If there was a spike, add an impulse response to the future features
                if S[t,n] > 0:
                    Xs[n][t+1:t+L+1, :] += self.basis.basis

            # Check Spike limit
            if np.any(S[t,:] >= max_spks_per_bin):
                n_exceptions += 1

            if np.any(S[t,:]>100):
                # raise Exception("More than 10 spikes in a bin! Decrease variance on impulse weights or decrease simulation bin width.")
                print "More than 10 spikes in a bin! Decrease variance on impulse weights or decrease simulation bin width."
                import pdb; pdb.set_trace()

        if verbose:
            print "Number of exceptions arising from multiple spikes per bin: %d" % n_exceptions

        if keep:
            Xs = [X[:T,:] for X in Xs]
            data = np.hstack(Xs + [S])
            self.add_data(data)

        return S, Xs

    def rvs(self,size=[]):
        return self.generate(keep=False, size=size)

    def plot_mean_spike_counts(self, Xs,
                               dt=None, t=None, S=None,
                               axs=None, lns=None,
                               color='b', style='-',
                               inds=None, t_lim=None):
        """
        Plot the mean spike counts for a given set of features.
        :param X:
        :param dt:
        :param t:
        :param S:
        :param axs:
        :param lns:
        :param color:
        :param style:
        :param inds:
        :param t_lim:
        :return:
        """
        N = self.N
        if inds is None:
            inds = np.arange(N)

        if t is None and dt is None:
            t = np.arange(Xs[0].shape[0])
            dt = 1
        elif t is None and dt is not None:
            t = np.arange(Xs[0].shape[0]) * dt
        else:
            dt = np.gradient(t)
        ymax = 1.0
        if lns is None and axs is None:
            lns = []
            axs = []
            fig = plt.figure()

            for i,n in enumerate(inds):
                ax = fig.add_subplot(len(inds),1,i+1)
                axs.append(ax)

                mu = self.neuron_models[n].mean(Xs) / dt
                print mu.mean(), " += ", mu.std()
                ln = ax.plot(t, mu, c=color, ls=style)
                lns.append(ln[0])

                # std = network.spike_train_models[n].std(fS)
                # plt.plot(mu+std, '--', c=color)

                if S is not None:
                    ymax = np.amax([ymax, np.amax(S), np.amax(mu)])
                else:
                    ymax = np.amax([ymax, np.amax(mu)+0.1])

            S_height = 1.1 * ymax

            for i,n in enumerate(inds):
                if S is not None:
                    t0 = np.nonzero(S[:,n])[0]
                    axs[i].plot(t[t0], S_height*np.ones_like(t0), 'ko', markerfacecolor='k', markersize=4)

                axs[i].set_ylim(0, 1.2*ymax)
                if t_lim is not None:
                    axs[i].set_xlim(t_lim)

        elif lns is None and axs is not None:
            lns = []
            for i,n in enumerate(inds):
                mu = self.neuron_models[n].mean(Xs) / dt
                ln = axs[i].plot(t, mu, c=color, ls=style)
                lns.append(ln[0])

                # std = network.spike_train_models[n].std(fS)
                # plt.plot(mu+std, '--', c=color)

                if S is not None:
                    ymax = np.amax([ymax, np.amax(S), np.amax(mu)])
                else:
                    ymax = np.amax([ymax, np.amax(mu)+0.1])

            ymax = 200/1.2
            S_height = 1.1 * ymax

            for i,n in enumerate(inds):
                if S is not None:
                    t0 = np.nonzero(S[:,n])[0]
                    axs[i].plot(t[t0], S_height*np.ones_like(t0), 'ko', markerfacecolor='k', markersize=4)

                axs[i].set_ylim(0, 1.2*ymax)
                if t_lim is not None:
                    axs[i].set_xlim(t_lim)

        else:
            # We are given handles, just update them
            for i,n in enumerate(inds):
                mu = self.neuron_models[n].mean(Xs) / dt
                lns[i].set_data(t, mu)

        return axs, lns

    def plot_weighted_network(self, ax=None, lns=None, cmap=plt.cm.RdGy, perm=None):
        A = self.A
        W = self.weights.sum(axis=-1)
        # Wmax = max(1, np.amax(abs(W)))
        Wmax = W.mean() + 3*W.std()

        Weff = A*W
        if perm is not None:
            Weff = permute_matrix(Weff, perm)

        print "N_conn: ", np.sum(A)
        print "W_mean: ", np.mean(A*W)

        if lns is None and ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            lns = ax.imshow(np.kron(Weff, np.ones((10,10))),
                            extent=[0,self.N, self.N, 0],
                            vmin=-Wmax, vmax=Wmax,
                            interpolation="none",
                            cmap=cmap)
            # cbax = fig.add
            ax.set_xlim([0,self.N])
            ax.set_ylim([self.N,0])
            ax.set_aspect('equal')

        elif lns is None and ax is not None:
            lns = ax.imshow(np.kron(Weff, np.ones((10,10))),
                            extent=[0,self.N, self.N, 0],
                            vmin=-Wmax, vmax=Wmax,
                            interpolation="none",
                            cmap=cmap)

        elif lns is not None and ax is not None:
            lns.set_data(np.kron(Weff, np.ones((10,10))))

        return ax, lns


class CompleteBernoulliPopulation(_PopulationOfNeuronsBase):
    def __init__(self, N,
                 basis,
                 neuron_hypers={},
                 network_hypers={},
                 global_bias_class=ScalarGaussianNIX,
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0}
                 ):
        super(CompleteBernoulliPopulation, self).\
            __init__(N,
                     basis=basis,
                     neuron_class=BernoulliNeuron,
                     neuron_hypers=neuron_hypers,
                     network_class=CompleteNetwork,
                     network_hypers=network_hypers,
                     global_bias_class=global_bias_class,
                     global_bias_hypers=global_bias_hypers
                     )


class CompleteNegativeBinomialPopulation(_PopulationOfNeuronsBase):
    def __init__(self, N,
                 basis,
                 neuron_hypers={},
                 network_hypers={},
                 global_bias_class=ScalarGaussianNIX,
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0}
                 ):
        super(CompleteNegativeBinomialPopulation, self).\
            __init__(N,
                     basis=basis,
                     neuron_class=NegativeBinomialNeuron,
                     neuron_hypers=neuron_hypers,
                     network_class=CompleteNetwork,
                     network_hypers=network_hypers,
                     global_bias_class=global_bias_class,
                     global_bias_hypers=global_bias_hypers
                     )


class ErdosRenyiNegativeBinomialPopulation(_PopulationOfNeuronsBase):
    def __init__(self, N,
                 basis,
                 neuron_hypers={},
                 network_hypers={},
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0}
                 ):
        super(ErdosRenyiNegativeBinomialPopulation, self).\
            __init__(N,
                     basis=basis,
                     neuron_class=NegativeBinomialSparseNeuron,
                     neuron_hypers=neuron_hypers,
                     network_class=ErdosRenyiNetwork,
                     network_hypers=network_hypers,
                     global_bias_hypers=global_bias_hypers
                     )

class ErdosRenyiBernoulliPopulation(_PopulationOfNeuronsBase):
    def __init__(self, N,
                 basis,
                 neuron_hypers={},
                 network_hypers={},
                 global_bias_class=ScalarGaussianNIX,
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0}
                 ):
        super(ErdosRenyiBernoulliPopulation, self).\
            __init__(N,
                     basis=basis,
                     neuron_class=BernoulliSparseNeuron,
                     neuron_hypers=neuron_hypers,
                     network_class=ErdosRenyiNetwork,
                     network_hypers=network_hypers,
                     global_bias_class=global_bias_class,
                     global_bias_hypers=global_bias_hypers
                     )


class _SBMPopulationBase(_PopulationOfNeuronsBase):
    def plot_sbm_weights(self, axs=None, lns=None, t=None, color='r'):
        """
        Plot the weight distribution of an SBM
        :return:
        """
        sbm = self.network
        K = sbm.K
        if t is None:
            t = np.arange(self.basis.L)

        if lns is None and axs is None:
            fig = plt.figure()
            axs = []
            for k1 in range(K):
                for k2 in range(K):
                    ax = fig.add_subplot(K,K,k1*K + k2 + 1)
                    axs.append(ax)

            lns = []
            for k1 in range(K):
                for k2 in range(K):
                    ax = axs[k1*K+ k2]
                    # ax.cla()
                    w_mu = sbm.weights_priors[k1][k2].mu
                    w_std = np.sqrt(np.diag(sbm.weights_priors[k1][k2].sigma))
                    imp = self.basis.basis.dot(w_mu)
                    impp1 = self.basis.basis.dot(w_mu +  w_std)
                    impm1 = self.basis.basis.dot(w_mu - w_std)
                    l1 = ax.plot(t, imp, '-', c=color)
                    l2 = ax.plot(t, impp1, '--', c=color)
                    l3 = ax.plot(t, impm1, '--', c=color)
                    ax.plot(np.zeros_like(imp), ':k')
                    lns.append((l1,l2,l3))

        elif lns is None and axs is not None:

            lns = []
            for k1 in range(K):
                for k2 in range(K):
                    ax = axs[k1*K+ k2]
                    # ax.cla()
                    w_mu = sbm.weights_priors[k1][k2].mu
                    w_std = np.sqrt(np.diag(sbm.weights_priors[k1][k2].sigma))
                    imp = self.basis.basis.dot(w_mu)
                    impp1 = self.basis.basis.dot(w_mu +  w_std)
                    impm1 = self.basis.basis.dot(w_mu)
                    l1 = ax.plot(t, imp, '-', c=color)
                    l2 = ax.plot(t, impp1, '--', c=color)
                    l3 = ax.plot(t, impm1, '--', c=color)
                    ax.plot(np.zeros_like(imp), ':k')
                    lns.append((l1,l2,l3))
        else:
            for k1 in range(K):
                for k2 in range(K):
                    l1,l2,l3 = lns[k1*K + k2]
                    w_mu = sbm.weights_priors[k1][k2].mu
                    w_std = np.sqrt(np.diag(sbm.weights_priors[k1][k2].sigma))
                    imp = self.basis.basis.dot(w_mu)
                    impp1 = self.basis.basis.dot(w_mu +  w_std)
                    impm1 = self.basis.basis.dot(w_mu - w_std)
                    l1[0].set_data(t,imp)
                    l2[0].set_data(t,impp1)
                    l3[0].set_data(t,impm1)


        return axs, lns



class SBMNegativeBinomialPopulation(_SBMPopulationBase):
    def __init__(self, N,
                 basis,
                 latent_variable_hypers={},
                 neuron_hypers={},
                 network_hypers={},
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0}
                 ):
        super(SBMNegativeBinomialPopulation, self).\
            __init__(N,
                     basis=basis,
                     latent_variable_class=LatentClass,
                     latent_variable_hypers=latent_variable_hypers,
                     neuron_class=NegativeBinomialSparseNeuron,
                     neuron_hypers=neuron_hypers,
                     network_class=StochasticBlockNetwork,
                     network_hypers=network_hypers,
                     global_bias_hypers=global_bias_hypers
                     )


class SBMBernoulliPopulation(_SBMPopulationBase):
    def __init__(self, N,
                 basis,
                 latent_variable_hypers={},
                 neuron_hypers={},
                 network_hypers={},
                 global_bias_hypers={'mu_0' : 0.0, 'kappa_0' : 1.0, 'sigmasq_0' : 0.1, 'nu_0' : 10.0}
                 ):
        super(SBMBernoulliPopulation, self).\
            __init__(N,
                     basis=basis,
                     latent_variable_class=LatentClass,
                     latent_variable_hypers=latent_variable_hypers,
                     neuron_class=BernoulliSparseNeuron,
                     neuron_hypers=neuron_hypers,
                     network_class=StochasticBlockNetwork,
                     network_hypers=network_hypers,
                     global_bias_hypers=global_bias_hypers
                     )
