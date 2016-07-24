import numpy as np
import os
import cPickle
import gzip
# np.seterr(all='raise')

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
    # Load some example data.
    # See data/synthetic/generate.py to create more.
    ###########################################################
    data_path = os.path.join("data", "synthetic", "synthetic_er_K5_T10000.pkl.gz")
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)

    # Load the test data
    test_path = os.path.join("data", "synthetic", "synthetic_er_K5_T10000_test.pkl.gz")
    with gzip.open(test_path, 'r') as f:
        S_test, _ = cPickle.load(f)

    T      = S.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            weight_hypers=true_model.weight_hypers,
                            network_hypers=true_model.network_hypers)
    test_model.add_data(S)

    # Convolve the test data for fast heldout likelihood calculations
    F_test = test_model.basis.convolve_with_basis(S_test)

    # Initialize plots
    ln, im_net = initialize_plots(true_model, test_model, S)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 100
    samples = []
    lps = []
    plls = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        plls.append(test_model.heldout_log_likelihood(S_test, F=F_test))
        samples.append(test_model.copy_sample())

        print ""
        print "Gibbs iteration ", itr
        print "LP: ", lps[-1]

        test_model.resample_model()

        # Update plot
        if itr % 5 == 0:
            update_plots(itr, test_model, S, ln, im_net)

    plt.ioff()

    ###########################################################
    # Analyze the samples
    ###########################################################
    analyze_samples(true_model, None, samples, lps, plls)

def initialize_plots(true_model, test_model, S):
    N = true_model.N
    true_model.add_data(S)
    W_lim = np.amax(abs(true_model.weight_model.W_effective.sum(2)))
    print "W_lim: ", W_lim
    R = true_model.compute_rate(true_model.data_list[0])
    T = S.shape[0]
    # Plot the true network
    plt.ion()
    plt.imshow(true_model.weight_model.W_effective.sum(2),
               vmax=W_lim, vmin=-W_lim,
               interpolation="none", cmap="RdGy")
    plt.colorbar()
    plt.pause(0.001)


    # Plot the true and inferred firing rate
    plt.figure(2)
    plt.plot(np.arange(T), R[:,0], '-k', lw=2)
    plt.ion()
    data = test_model.data_list[0]
    ln = plt.plot(np.arange(T), test_model.compute_rate(data)[:,0], '-r')[0]
    plt.show()

    # # Plot the block affiliations
    # plt.figure(3)
    # KC = np.zeros((K,C))
    # KC[np.arange(K), test_model.network.c] = 1.0
    # im_clus = plt.imshow(KC,
    #                 interpolation="none", cmap="Greys",
    #                 aspect=float(C)/K)
    #
    plt.figure(4)
    im_net = plt.imshow(test_model.weight_model.W_effective.sum(2),
                        vmax=W_lim, vmin=-W_lim,
                        interpolation="none", cmap="RdGy")
    plt.colorbar()
    plt.pause(0.001)

    plt.show()
    plt.pause(0.001)

    # return ln, im_net, im_clus
    return ln, im_net

def update_plots(itr, test_model, S, ln, im_net):
    N = test_model.N
    T = S.shape[0]
    plt.figure(2)
    data = test_model.data_list[0]
    ln.set_data(np.arange(T), test_model.compute_rate(data)[:,0])
    plt.title("\lambda_{%d}. Iteration %d" % (0, itr))
    plt.pause(0.001)

    # plt.figure(3)
    # KC = np.zeros((K,C))
    # KC[np.arange(K), test_model.network.c] = 1.0
    # im_clus.set_data(KC)
    # plt.title("KxC: Iteration %d" % itr)
    # plt.pause(0.001)

    plt.figure(4)
    plt.title("W: Iteration %d" % itr)
    im_net.set_data(test_model.weight_model.W_effective.sum(2))
    plt.pause(0.001)

def analyze_samples(true_model, init_model, samples, lps, plls):
    N_samples = len(samples)
    # Compute sample statistics for second half of samples
    A_samples = np.array([s.weight_model.A for s in samples])
    W_samples = np.array([s.weight_model.W for s in samples])
    b_samples = np.array([s.bias_model.b   for s in samples])
    p_samples = np.array([s.network.p      for s in samples])
    lps       = np.array(lps)

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    b_mean       = b_samples[offset:, ...].mean(axis=0)
    p_mean       = p_samples[offset:, ...].mean(axis=0)


    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "b true:        ", true_model.bias_model.b
    print ""
    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "b mean:        ", b_mean
    print "p mean:        ", p_mean

    plt.figure()
    plt.plot(np.arange(N_samples), lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    # # Predictive log likelihood
    # pll_init = init_model.heldout_log_likelihood(S_test)
    plt.figure()
    # plt.plot(np.arange(N_samples), pll_init * np.ones(N_samples), 'k')
    plt.plot(np.arange(N_samples), plls, 'r')
    plt.xlabel("Iteration")
    plt.ylabel("Predictive log probability")
    plt.show()

    # Compute the link prediction accuracy curves
    # auc_init = roc_auc_score(true_model.weight_model.A.ravel(),
    #                          init_model.W.ravel())
    # auc_A_mean = roc_auc_score(true_model.weight_model.A.ravel(),
    #                            A_mean.ravel())
    # auc_W_mean = roc_auc_score(true_model.weight_model.A.ravel(),
    #                            W_mean.ravel())
    #
    # aucs = []
    # for A in A_samples:
    #     aucs.append(roc_auc_score(true_model.weight_model.A.ravel(), A.ravel()))
    #
    # plt.figure()
    # plt.plot(aucs, '-r')
    # plt.plot(auc_A_mean * np.ones_like(aucs), '--r')
    # plt.plot(auc_W_mean * np.ones_like(aucs), '--b')
    # plt.plot(auc_init * np.ones_like(aucs), '--k')
    # plt.xlabel("Iteration")
    # plt.ylabel("Link prediction AUC")
    # plt.show()
    #

    # Compute the adjusted mutual info score of the clusterings
    # amis = []
    # arss = []
    # for c in c_samples:
    #     amis.append(adjusted_mutual_info_score(true_model.network.c, c))
    #     arss.append(adjusted_rand_score(true_model.network.c, c))
    #
    # plt.figure()
    # plt.plot(np.arange(N_samples), amis, '-r')
    # plt.plot(np.arange(N_samples), arss, '-b')
    # plt.xlabel("Iteration")
    # plt.ylabel("Clustering score")


demo(11223344)
