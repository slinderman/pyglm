"""
Network models that underlie the collection of negative binomial spike trains
"""
import numpy as np

from deps.pybasicbayes.abstractions import GibbsSampling
from deps.pybasicbayes.distributions import Gaussian, GaussianFixedCov, GaussianFixedMean


class _NetworkBase(GibbsSampling):
    """
    Base class for various network models.
    """
    def __init__(self, population):
        self.population = population
        self.N = self.population.N

    def weights_prior(self, n_pre, n_post):
        return None

    # def update_spike_train_priors(self):
    #     # Update the regression model priors
    #     for n_pre in range(self.N):
    #         for n_post in range(self.N):
    #             stm = self.population.spike_train_models[n_post]
    #             stm.model.regression_models[n_pre].weights_prior = \
    #                 self.weights_prior(n_pre, n_post)

    @property
    def parameters(self):
        return ()



class ErdosRenyiNetwork(_NetworkBase):
    """
    Simplest nontrivial network model you can think of.
    Each connection is an iid Bernoulli r.v. with parameter rho.
    The weights of the edges are also i.i.d.
    """
    def __init__(self, population,
                 rho=None,
                 weight_prior_class=Gaussian,
                 weight_prior_hypers={},
                 refractory_rho=None,
                 refractory_prior_class=Gaussian,
                 refractory_prior_hypers={}
                ):

        super(ErdosRenyiNetwork, self).__init__(population)

        # Inititalize the network
        self.rho = rho
        if self.rho is None:
            # Sample rho from a beta prior before sampling A
            self.rho = np.random.beta(1.0, 1.0) * np.ones((self.N, self.N))
        elif np.isscalar(self.rho) or self.rho.size == 1:
            self.rho = float(self.rho) * np.ones((self.N, self.N))
        else:
            assert self.rho.shape == (self.N, self.N), "Rho must be NxN or scalar!"

        # Set the refractory sparsity level
        if refractory_rho is not None:
            self.rho[np.diag_indices(self.N)] = refractory_rho

        # Instantiate the prior over the weights
        self.syn_prior = weight_prior_class(**weight_prior_hypers)

        # Instantiate the prior over the weights
        self.refractory_prior = refractory_prior_class(**refractory_prior_hypers)

    @property
    def parameters(self):
        # return self.refractory_prior, self.syn_prior
        return self.refractory_prior.mu, \
               self.refractory_prior.sigma, \
               self.syn_prior.mu, \
               self.syn_prior.sigma

    @parameters.setter
    def parameters(self, (rmu, rsigma, smu, ssigma)):
        self.refractory_prior.mu = rmu
        self.refractory_prior.sigma = rsigma
        self.syn_prior.mu = smu
        self.syn_prior.sigma = ssigma

    @property
    def parameter_names(self):
        return "refractory mean", "refractory cov", "syn mean", "syn cov"

    def weights_prior(self, n_pre, n_post):
        if n_pre == n_post:
            return self.refractory_prior
        else:
            return self.syn_prior

    def rvs(self,size=1):
        """
        Sample a network (pair of adjacency matrix and weights)
        :param size: integer number of samples
        :return:
        """
        assert isinstance(size, int)
        N = self.N
        samples = []
        for i in range(size):
            A = np.random.rand(N,N) < self.rho
            W = self.syn_prior.rvs(size=[N,N])
            samples.append((A,W))

        if size == 1:
            return samples[0]
        else:
            return samples


    def log_likelihood(self, x=None):
        if x is not None:
            A,W = x
        else:
            A,W = self.population.A, self.population.weights

        ll = 0
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                if A[n_pre, n_post]:
                    ll += np.log(self.rho[n_pre, n_post]) * \
                          self.syn_prior.log_likelihood(W[n_pre,n_post,:])
                else:
                    ll += np.log(1-self.rho[n_pre, n_post])

    def resample(self,data=[]):
        if isinstance(data, list):
            As = [A for A,W in data]
            Ws = [W for A,W in data]

        else:
            assert isinstance(data, tuple)
            As = [data[0]]
            Ws = [data[1]]

        # Separate out "synaptic" and "refractory" weights
        W_syn = []
        W_ref = []
        for A,W in zip(As,Ws):
            for n_pre in range(self.N):
                for n_post in range(self.N):
                    if A[n_pre, n_post]:
                        if n_pre == n_post:
                            W_ref.append(W[n_pre,n_post,:])
                        else:
                            W_syn.append(W[n_pre,n_post,:])

        # Convert lists to np arrays
        if len(W_syn) > 0:
            W_syn = np.array(W_syn).reshape((len(W_syn),-1))
        if len(W_ref) > 0:
            W_ref = np.array(W_ref).reshape((len(W_ref),-1))

        self.syn_prior.resample(W_syn)
        self.refractory_prior.resample(W_ref)

        # Update the regression model priors
        # self.update_spike_train_priors()

class StochasticBlockNetwork(_NetworkBase):
    """
    Stochastic block network: neurons have a latent class, c_n.
    Each pair of classes is associated with a connection probability,
        \rho_{c_{n'} \to c_{n}}
    and a weight distribution
        p(w_{n' \to n} | c_n', c_n, ...)

    The number of classes is K.

    """
    def __init__(self, population,
                 rho=None,
                 weight_prior_hypers={},
                 refractory_rho=None,
                 refractory_prior_hypers={}
                ):

        super(StochasticBlockNetwork, self).__init__(population)

        # The class_model keeps track of the latent type of each neuron
        # and calls into this distribution to evaluate the probability of
        # a latent class assignment.
        self.class_model = self.population.latent

        # Inititalize the network
        self.rho = rho
        if self.rho is None:
            # Sample rho from a beta prior before sampling A
            self.rho = np.random.beta(1.0, 1.0) * np.ones((self.N, self.N))
        elif np.isscalar(self.rho) or self.rho.size == 1:
            self.rho = float(self.rho) * np.ones((self.N, self.N))
        else:
            assert self.rho.shape == (self.N, self.N), "Rho must be NxN or scalar!"

        # Learn a joint prior over the variance of the weights
        self.B = weight_prior_hypers['mu_0'].size
        self.weight_variance_model  = GaussianFixedMean(mu=np.zeros(1,),
                                                        # sigma=0.1*np.eye(1),
                                                        nu_0=1,
                                                        lmbda_0=0.01*np.eye(1))

        Sigma = np.asscalar(self.weight_variance_model.getsigma()) * np.eye(self.B)

        # Set the refractory sparsity level
        if refractory_rho is not None:
            self.rho[np.diag_indices(self.N)] = refractory_rho

        # Instantiate the prior over the weights
        self.weights_priors = [[GaussianFixedCov(sigma=Sigma,
                                                 **weight_prior_hypers)
                                for c1 in range(self.K)]
                                 for c2 in range(self.K)]

        # Instantiate the prior over the weights
        self.refractory_prior = GaussianFixedCov(sigma=Sigma,
                                                 **refractory_prior_hypers)

    @property
    def K(self):
        return self.class_model.K

    @property
    def classes(self):
        return self.class_model.classes

    def weights_prior(self, n_pre, n_post):
        c_pre = self.classes[n_pre]
        c_post = self.classes[n_post]
        if n_pre == n_post:
            return self.refractory_prior
        else:
            return self.weights_priors[c_pre][c_post]

    def smart_initialize(self):
        """
        Initialize by clustering the weights
        :return:
        """
        from sklearn.cluster import KMeans
        W_eff = self.population.W_effective
        features = []
        for n in xrange(self.N):
            features.append(np.concatenate((W_eff[:,n], W_eff[n,:])))

        self.class_model.classes = KMeans(n_clusters=self.K).fit(np.array(features)).labels_
        print "Smart classes: ", self.class_model.classes
        self.resample((self.population.A, self.population.weights))

    def rvs(self,size=1):
        """
        Sample a network (pair of adjacency matrix and weights)
        :param size: integer number of samples
        :return:
        """
        assert isinstance(size, int)
        N = self.N
        c = self.classes
        samples = []
        for i in range(size):
            # import pdb; pdb.set_trace()

            A = np.random.rand(N,N) < self.rho
            W = np.zeros((N,N,self.weights_priors[0].D))
            for n_pre in xrange(N):
                for n_post in xrange(N):
                    if n_pre == n_post:
                        W[n_pre, n_post,:] = self.refractory_prior.rvs(1)
                    else:
                        W[n_pre, n_post,:] = self.weights_priors[c[n_pre]][c[n_post]].rvs(size=1)
            samples.append((A,W))

        if size == 1:
            return samples[0]
        else:
            return samples


    def log_likelihood(self, x=None):
        if x is not None:
            A,W = x
        else:
            A,W = self.population.A, self.population.weights
        ll = 0
        for n_pre in xrange(self.N):
            for n_post in xrange(self.N):
                if A[n_pre, n_post]:
                    tmp = np.log(self.rho[n_pre, n_post]) * \
                          self.weights_prior(n_pre, n_post).\
                              log_likelihood(W[n_pre,n_post,:])
                    ll += tmp
                    # print "LL %d->%d: %f" % (n_pre, n_post, tmp)
                else:
                    ll += np.log(1-self.rho[n_pre, n_post])

        return ll


    def resample(self,data=[]):
        if isinstance(data, list):
            As = [A for A,W in data]
            Ws = [W for A,W in data]

        else:
            assert isinstance(data, tuple)
            As = [data[0]]
            Ws = [data[1]]

        self.resample_weight_means(As, Ws)
        self.resample_weight_variance(As, Ws)
        # TODO: Sample connection probability

    def resample_weight_means(self,As, Ws):
        # Sample prior over "synaptic" weights per block
        for c_pre in xrange(self.K):
            for c_post in xrange(self.K):
                I_pre = np.where(self.classes == c_pre)[0]
                I_post = np.where(self.classes == c_post)[0]

                W_syn = []
                for A,W in zip(As,Ws):
                    for n_pre in I_pre:
                        for n_post in I_post:
                            if A[n_pre, n_post]:
                                if n_pre == n_post: pass
                                else:
                                    W_syn.append(W[n_pre,n_post,:])

                # Convert lists to np arrays
                if len(W_syn) > 0:
                    W_syn = np.array(W_syn).reshape((len(W_syn),-1))
                    # import pdb; pdb.set_trace()

                # Resample
                self.weights_priors[c_pre][c_post].resample(W_syn)
                if len(W_syn) > 0:
                    # DEBUG
                    print "Block %d->%d" % (c_pre, c_post)
                    print "n: ", len(W_syn)
                    print "Mean %d->%d: %s " % (c_pre, c_post, str(self.weights_priors[c_pre][c_post].mu))
                    print "Mean W_syn: ", W_syn.mean(0)
                    print "Std %d->%d: %s " % (c_pre, c_post, str(np.diag(np.sqrt(self.weights_priors[c_pre][c_post].sigma))))
                    print "Std W_syn: ", W_syn.std(0)


        # Sample refractory weights
        W_ref = []
        for A,W in zip(As,Ws):
            for n in xrange(self.N):
                if A[n,n]:
                    W_ref.append(W[n,n,:])

        if len(W_ref) > 0:
            W_ref = np.array(W_ref).reshape((len(W_ref),-1))
        self.refractory_prior.resample(W_ref)

        # self.update_spike_train_priors()

    def resample_weight_variance(self, As, Ws):
        """
        Resample the weight variance shared by all neurons

        :return:
        """
        # Extract the weights (don't differentiate between refractory and synaptic)
        Weffs = []
        for A,W in zip(As,Ws):
            for n_pre in xrange(self.N):
             for n_post in xrange(self.N):
                if A[n_pre, n_post] and n_pre != n_post:
                    Weffs.append(W[n_pre, n_post,:] - self.weights_prior(n_pre, n_post).mu)

        Weffs = np.array(Weffs)
        self.weight_variance_model.resample(Weffs.ravel())

        # Set the variance for the weight priors
        Sigma = self.weight_variance_model.sigma[0,0] * np.eye(self.B)
        for c_pre in xrange(self.K):
            for c_post in xrange(self.K):
                self.weights_priors[c_pre][c_post].setsigma(Sigma)
