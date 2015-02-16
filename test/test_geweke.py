from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot, invgamma

from pyglm.populations import *
from pyglm.deps.pybasicbayes.distributions import DiagonalGaussian, GaussianFixed


seed = np.random.randint(2**16)
print "Setting random seed to ", seed
np.random.seed(seed)

################
#  parameters  #
################
def create_simple_population(alpha_0=10.0, beta_0=10.0,
                             mu_bias=-3.0, sigma_bias=0.5**2,
                             mu_w=-0.5, sigma_w=0.5**2,
                             rho=0.5):
    N = 1
    dt = 0.001
    T = 100

    # Set the model parameters
    B = 1       # Number of basis functions
    neuron_hypers = {'alpha_0' : alpha_0,
                     'beta_0' : beta_0}

    global_bias_hypers= {'mu' : mu_bias,
                         'sigmasq' : sigma_bias}

    network_hypers = {'rho' : rho,
                      'weight_prior_class' : DiagonalGaussian,
                      'weight_prior_hypers' :
                          {
                              'mu' : mu_w * np.ones((B,)),
                              'sigmas' : sigma_w * np.ones(B)
                          },

                      'refractory_rho' : rho,
                      'refractory_prior_class' : DiagonalGaussian,
                      'refractory_prior_hypers' :
                          {
                              'mu' : mu_w * np.ones((B,)),
                              'sigmas' : sigma_w * np.ones(B)
                          },
                     }

    population = ErdosRenyiBernoulliPopulation(
            N, B=B, dt=dt,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=neuron_hypers,
            network_hypers=network_hypers,
            )

    population.generate(size=T, keep=True)
    return population

def test_bias_geweke(N_samples=100000, thin=1):
    mu_bias = -3.0
    sigma_bias = 0.5**2
    population = create_simple_population(mu_bias=mu_bias, sigma_bias=sigma_bias)

    bias_samples = []
    for s in xrange(N_samples):
        print "Iteration: ", s
        for _ in xrange(thin):
            population.resample_model(do_resample_bias=True,
                                      do_resample_bias_prior=False,
                                      do_resample_latent=False,
                                      do_resample_network=False,
                                      do_resample_sigma=False,
                                      do_resample_synapses=False,
                                      do_resample_psi=False,
                                      do_resample_psi_from_prior=True)

        # Collect samples
        bias_samples.append(population.biases.copy())

    # Convert samples to arrays
    bias_samples = np.array(bias_samples)

    bias_mean = bias_samples.mean(0)
    bias_std = bias_samples.std(0)
    bias_dist = norm(mu_bias, np.sqrt(sigma_bias))
    print "Mean bias: ", bias_mean, " +- ", bias_std
    # Make Q-Q plots
    fig = plt.figure()
    bias_ax = fig.add_subplot(121)
    probplot(bias_samples[:,0,0], dist=bias_dist, plot=bias_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(bias_samples[:,0,0], 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, bias_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

def test_bias_geweke(N_samples=100000, thin=1):
    mu_bias = -3.0
    sigma_bias = 0.5**2
    population = create_simple_population(mu_bias=mu_bias, sigma_bias=sigma_bias)

    bias_samples = []
    for s in xrange(N_samples):
        print "Iteration: ", s
        for _ in xrange(thin):
            population.resample_model(do_resample_bias=True,
                                      do_resample_bias_prior=False,
                                      do_resample_latent=False,
                                      do_resample_network=False,
                                      do_resample_sigma=False,
                                      do_resample_synapses=False,
                                      do_resample_psi=False,
                                      do_resample_psi_from_prior=True)

        # Collect samples
        bias_samples.append(population.biases.copy())

    # Convert samples to arrays
    bias_samples = np.array(bias_samples)

    bias_mean = bias_samples.mean(0)
    bias_std = bias_samples.std(0)
    bias_dist = norm(mu_bias, np.sqrt(sigma_bias))
    print "Mean bias: ", bias_mean, " +- ", bias_std
    # Make Q-Q plots
    fig = plt.figure()
    bias_ax = fig.add_subplot(121)
    probplot(bias_samples[:,0,0], dist=bias_dist, plot=bias_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(bias_samples[:,0,0], 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, bias_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

def test_weights_geweke(N_samples=200000, thin=1):
    mu_w = 0.0
    sigma_w = 0.5**2
    rho = 0.5
    population = create_simple_population(mu_w=mu_w, sigma_w=sigma_w, rho=rho)

    A_samples = []
    w_samples = []
    for s in xrange(N_samples):
        print "Iteration: ", s
        # Resampling is trickier because of the augmentation.
        for _ in xrange(thin):
            population.resample_model(do_resample_bias=False,
                                      do_resample_bias_prior=False,
                                      do_resample_latent=False,
                                      do_resample_network=False,
                                      do_resample_sigma=False,
                                      do_resample_synapses=True,
                                      do_resample_psi=False,
                                      do_resample_psi_from_prior=True)

        # Collect samples
        A_samples.append(population.A.copy())
        w_samples.append(population.weights.copy())

    # Convert samples to arrays
    assert population.N == 1
    A_samples = np.array(A_samples).ravel()
    A_mean = A_samples.mean(0)
    print "Mean A: \n", A_mean

    # Get the samples where A is nonzero
    w_samples = np.array(w_samples)[A_samples > 0, ...]
    w_mean = w_samples.mean(0)
    w_std = w_samples.std(0)
    print "Mean w: \n", w_mean, " +- ", w_std


    # Make Q-Q plots
    fig = plt.figure()
    w_ax = fig.add_subplot(121)
    w_dist = norm(mu_w, np.sqrt(sigma_w))
    probplot(w_samples[:,0,0,0], dist=w_dist, plot=w_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(w_samples[:,0,0,0], 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, w_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

def test_sigma_geweke(N_samples=100000, thin=1):
    alpha_0 = 3.0
    beta_0 = 0.5
    population = create_simple_population(alpha_0=alpha_0,
                                          beta_0=beta_0)

    sigma_samples = []
    for s in xrange(N_samples):
        print "Iteration: ", s
        # Resampling is trickier because of the augmentation.
        for _ in xrange(thin):
            population.resample_model(do_resample_bias=False,
                                      do_resample_bias_prior=False,
                                      do_resample_latent=False,
                                      do_resample_network=False,
                                      do_resample_sigma=True,
                                      do_resample_synapses=False,
                                      do_resample_psi=False,
                                      do_resample_psi_from_prior=True)

        # Collect samples
        sigma_samples.append(population.etas.copy())

    # Convert samples to arrays
    sigma_samples = np.array(sigma_samples)
    sigma_mean = sigma_samples.mean(0)
    sigma_std = sigma_samples.std(0)
    print "Mean sigma: \n", sigma_mean, " +- ", sigma_std

    # Make Q-Q plots
    fig = plt.figure()
    # w_ax = fig.add_subplot(121)
    sigma_dist = invgamma(a=alpha_0, scale=beta_0)
    # probplot(w_samples[:,0,0,0], dist=w_dist, plot=w_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(sigma_samples[:,0], 50, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, sigma_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

def test_polya_gamma_geweke(N_samples=10000, thin=1, T=1):
    """
    Test the PolyaGamma augmentation with geweke sampling

    :return:
    """
    from pyglm.internals.observations import AugmentedBernoulliCounts

    mu_psi = 0.0
    sigma_psi = 1.0
    class DummyNeuron:
        sigma = np.array(sigma_psi)
        def mean_activation(self, X):
            return mu_psi * np.ones(T)

    # Make a counts object
    neuron = DummyNeuron()
    X = np.zeros(T)
    S = np.zeros(T)
    counts = AugmentedBernoulliCounts(X, S, neuron)
    S = counts.rvs()
    counts.counts = S

    psi_samples = []
    for s in xrange(N_samples):
        print "Iteration ", s
        # Resampling is trickier because of the augmentation.
        for _ in xrange(thin):
            # Resample psi and omega given counts
            counts.resample()

            # Resample counts given psi
            S = counts.rvs()
            counts.counts = S

        # Collect samples
        psi_samples.append(counts.psi.copy())

    # Convert samples to arrays
    psi_samples = np.array(psi_samples)
    psi_mean = psi_samples.mean(0)
    psi_std = psi_samples.std(0)
    print "Mean psi: \n", psi_mean, " +- ", psi_std

    # Make Q-Q plots
    fig = plt.figure()
    psi_ax = fig.add_subplot(121)
    psi_dist = norm(mu_psi, np.sqrt(sigma_psi))
    probplot(psi_samples[:,0], dist=psi_dist, plot=psi_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(psi_samples[:,0], 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, psi_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()




# test_bias_geweke()
test_weights_geweke(N_samples=100000)
# test_sigma_geweke(N_samples=100000)
# test_polya_gamma_geweke()


