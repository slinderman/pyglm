import numpy as np
import os
import cPickle
import gzip
import time

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.models import Population, StandardBernoulliPopulation

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
    # Load the RGC data
    ###########################################################
    base_path = os.path.join("data", "rgc", "rgc_60T")
    data_path = base_path + ".pkl"
    with open(data_path, 'r') as f:
        data = cPickle.load(f)

    # Set / extract model parameters
    S      = data["S"].astype(np.int32)
    T      = data["T"]
    N      = data["N"]
    dt     = data["dt"]
    B      = 3
    dt_max = 0.100

    # Load the initial model (fit via L1-logistic regression)
    init_path = base_path + ".standard_fit.l1.pkl.gz"
    with gzip.open(init_path, 'r') as f:
        init_model = cPickle.load(f)

    # Load the test data
    test_path = os.path.join("data", "rgc", "rgc_test.pkl")
    with open(test_path, 'r') as f:
        test_data = cPickle.load(f)
        S_test = test_data["S"].astype(np.int32)

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################
    # Use the initial model to set hypers
    bias_hypers = {"mu_0": init_model.bias.mean(),
                   "sigma_0": init_model.bias.var()}

    network_hypers = {"mu_0": init_model.W.mean(axis=(0,1)),
                      "Sigma_0": np.diag(init_model.W.var(axis=(0,1)))}

    # Copy the network hypers.
    test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            bias_hypers=bias_hypers,
                            network_hypers=network_hypers)
    test_model.add_data(S)

    # Initialize the test model parameters with the
    # parameters of the L1-regularized logistic regression model
    test_model.initialize_with_standard_model(init_model)

    # Convolve the test data for fast heldout likelihood calculations
    F_test = test_model.basis.convolve_with_basis(S_test)

    # Initialize plots
    lns, im_net = initialize_plots(test_model, S)

    ###########################################################
    # Fit the test model with batch variational inference
    ###########################################################
    N_iters = 20
    plot_interval = np.inf
    samples = [test_model.copy_sample()]
    lps = [test_model.log_probability()]
    plls = [test_model.heldout_log_likelihood(S_test, F=F_test)]

    timestamps = [0]
    start = time.clock()
    for itr in xrange(N_iters):
        print ""
        print "Gibbs iteration ", itr
        test_model.resample_model()

        samples.append(test_model.copy_sample())
        plls.append(test_model.heldout_log_likelihood(S_test, F=F_test))
        lps.append(test_model.log_probability())
        timestamps.append(time.clock()-start)
        print "LP:            ", lps[-1]
        print "PLL:           ", plls[-1]

        # Update plot
        if itr % plot_interval == 0:
            update_plots(itr, test_model, S, lns, im_net)
    plt.ioff()

    ###########################################################
    # Analyze the samples
    ###########################################################
    analyze_samples(init_model, samples, lps, plls, S_test)

    ###########################################################
    # Save the results
    ###########################################################
    results_path = base_path + ".er_fit.gibbs.pkl.gz"
    with gzip.open(results_path, 'w') as f:
        cPickle.dump((samples, lps, plls, timestamps), f, protocol=-1)

def initialize_plots(test_model, S):
    N = test_model.N
    W_lim = np.amax(abs(test_model.weight_model.W_effective.sum(2)))
    T = S.shape[0]

    plt.ion()
    plt.figure(1)
    im_net = plt.imshow(test_model.weight_model.mf_expected_W().sum(2),
                        vmax=W_lim, vmin=-W_lim,
                        interpolation="none", cmap="RdGy")
    plt.colorbar()
    plt.pause(0.001)

    # Plot the true and inferred firing rate
    lns = []
    plt.figure(2)
    N_plot = min(N, 5)
    for n in xrange(N_plot):
        plt.subplot(N_plot,1,n+1)
        data = test_model.data_list[0]
        lns.append(plt.plot(np.arange(T),
                            test_model.mf_expected_rate(data)[:,n],
                            '-r')[0])
        # plt.ylim([0,1])
    plt.show()
    plt.pause(0.001)

    return lns, im_net

def update_plots(itr, test_model, S, lns, im_net):
    N = test_model.N
    T = S.shape[0]

    plt.figure(1)
    plt.title("W: Iteration %d" % itr)
    im_net.set_data(test_model.weight_model.mf_expected_W().sum(2))
    plt.pause(0.001)

    plt.figure(2)
    data = test_model.data_list[0]
    N_plot = min(N, 5)
    for n in xrange(N_plot):
        plt.subplot(N_plot,1,n+1)
        lns[n].set_data(np.arange(T), test_model.mf_expected_rate(data)[:,n])
    plt.title("Iteration %d" % (itr))
    plt.pause(0.001)

def analyze_samples(init_model, samples, lps, plls, S_test):
    N_samples = len(samples)
    # Compute sample statistics for second half of samples
    A_samples = np.array([s.weight_model.A     for s in samples])
    W_samples = np.array([s.weight_model.W     for s in samples])
    b_samples = np.array([s.bias_model.b       for s in samples])
    p_samples = np.array([s.network.p          for s in samples])
    lps      = np.array(lps)

    offset = N_samples // 2
    A_mean = A_samples[offset:, ...].mean(axis=0)
    W_mean = W_samples[offset:, ...].mean(axis=0)
    b_mean = b_samples[offset:, ...].mean(axis=0)
    p_mean = p_samples[offset:, ...].mean(axis=0)

    plt.figure()
    plt.plot(np.arange(N_samples), lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log Probability")

    # Predictive log likelihood
    pll_init = init_model.heldout_log_likelihood(S_test)
    plt.figure()
    plt.plot(np.arange(N_samples), pll_init * np.ones(N_samples), 'k')
    plt.plot(np.arange(N_samples), plls, 'r')
    plt.xlabel("Iteration")
    plt.ylabel("Predictive log probability")
    plt.show()


demo(1234)
