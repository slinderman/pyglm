import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, probplot, invgamma

from pybasicbayes.util.text import progprint_xrange

from pyglm.models import Population

def demo(seed=None):
    """
    Fit a weakly sparse
    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################
    N = 2                                                   # Number of neurons
    T = 100                                                 # Number of time bins
    dt = 1.0                                                # Time bin width
    dt_max = 10.0                                           # Max time of synaptic influence
    B = 2                                                   # Number of basis functions for the weights

    # Test the model at a particular temperature
    temperature = 0.25

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -1.0, "sigma_0": 0.25}

    p = 0.5                 # Probability of connection for each pair of clusters
    mu = np.zeros((B,))     # Mean weight for each pair of clusters
    sigma = 1.0 * np.eye(B) # Covariance of weight for each pair of clusters

    ###########################################################
    # Create the model with these parameters
    ###########################################################
    network_hypers = {'p': p, 'mu': mu, 'sigma': sigma}
    # Copy the network hypers.
    test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            bias_hypers=bias_hypers,
                            network_hypers=network_hypers)

    # Sample some initial data
    test_model.generate(T=T, keep=True, verbose=False, temperature=temperature)

    ###########################################################
    # Run the Geweke test
    ###########################################################
    N_samples = 1000
    samples = []
    lps = []
    import ipdb; ipdb.set_trace()
    for itr in progprint_xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())

        # Resample the model given the data
        test_model.collapsed_resample_model(temperature=temperature)

        # Remove the old data and sample new
        test_model.data_list.pop()
        test_model.generate(T=T, keep=True, verbose=False, temperature=temperature)

        # Resample the observation model to ensure the auxiliary
        # variables are in sync with the new spike counts.
        # Since the observation model is resampled first, this
        # will be done first in the next call to resample_model().
        # test_model.observation_model.resample(test_model.data_list[0])

    ###########################################################
    # Check that the samples match the prior
    ###########################################################
    check_bias_samples(test_model, samples)
    check_weight_samples(test_model, samples)

def check_bias_samples(test_model, samples):
    """
    Check that the bias samples match the prior
    :param test_model:
    :param samples:
    :return:
    """
    mu_bias = test_model.bias_model.mu_0
    sigma_bias = test_model.bias_model.sigma_0

    # Convert samples to arrays
    bias_samples = np.array([s.bias_model.b for s in samples])

    bias_mean = bias_samples.mean(0)
    bias_std = bias_samples.std(0)
    bias_dist = norm(mu_bias, np.sqrt(sigma_bias))
    print "Mean bias: ", bias_mean, " +- ", bias_std
    # Make Q-Q plots
    fig = plt.figure()
    bias_ax = fig.add_subplot(121)
    probplot(bias_samples[:,0], dist=bias_dist, plot=bias_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(bias_samples[:,0], 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, bias_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

def check_weight_samples(test_model, samples):
    mu_w = test_model.network.weights.Mu
    Sigma_w = test_model.network.weights.Sigma
    rho = test_model.network.adjacency.P

    A_samples = np.array([s.weight_model.A for s in samples])
    W_samples = np.array([s.weight_model.W for s in samples])

    # Check that A's mean is about p
    A_mean = A_samples.mean(0)
    print "P:        ", rho
    print "Mean A: \n", A_mean

    # Get the samples where A is nonzero
    # assert test_model.N == 1
    n_pre = n_post = b = 1
    w_samples = np.array(W_samples[:,n_pre, n_post, b])[A_samples[:, n_pre, n_post] > 0, ...]
    w_mean = w_samples.mean(0)
    w_std = w_samples.std(0)
    print "Mean w: \n", w_mean, " +- ", w_std


    # Make Q-Q plots
    fig = plt.figure()
    w_ax = fig.add_subplot(121)
    w_dist = norm(mu_w[n_pre, n_post, b], np.sqrt(Sigma_w[n_pre, n_post, b, b]))
    probplot(w_samples, dist=w_dist, plot=w_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(w_samples, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, w_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

def check_eta_samples(test_model, samples):
    from pyglm.internals.activation import GaussianNoiseActivation
    if not isinstance(test_model.activation_model, GaussianNoiseActivation):
        return
    alpha_eta = test_model.activation_model.alpha_0
    beta_eta  = test_model.activation_model.beta_0

    # Convert samples to arrays
    eta_samples = np.array([s.activation_model.eta for s in samples])

    eta_mean = eta_samples.mean(0)
    eta_std  = eta_samples.std(0)
    eta_dist = invgamma(a=alpha_eta, scale=beta_eta)

    # Make Q-Q plots
    fig = plt.figure()
    # w_ax = fig.add_subplot(121)
    # probplot(w_samples[:,0,0,0], dist=w_dist, plot=w_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(eta_samples[:,0], 50, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, eta_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()



demo(1234)
