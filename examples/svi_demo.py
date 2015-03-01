import numpy as np
import os
import cPickle
import gzip
# np.seterr(all='raise')

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.models import Population, StandardBernoulliPopulation
from pyglm.utils.experiment_helper import load_data, load_results

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
    dataset = "synth_nb_eigen_K50_T10000"
    run = 0.001
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    assert os.path.exists(res_dir), "Results directory does not exist: " + res_dir
    standard_results = load_results(dataset, run=run,
                                    algorithms=["bfgs"])

    T      = train.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    # Create and fit a standard model for initialization
    init_model = standard_results["bfgs"]

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            bias_hypers=true_model.bias_hypers,
                            weight_hypers=true_model.weight_hypers,
                            network_hypers=true_model.network_hypers)

    # Add the data in minibatches
    test_model.add_data(S, minibatchsize=1000)
    test_model.initialize_with_standard_model(init_model)
    # F_test = test_model.basis.convolve_with_basis(S_test)

    # Initialize plots
    lns, im_net = initialize_plots(true_model, test_model, S)

    ###########################################################
    # Fit the test model with stochastic variational inference (SVI)
    ###########################################################
    # Stochastic variational inference
    N_iters = 10000
    samples = []
    delay = 1.0
    forgetting_rate = 0.25
    stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
    raw_input("Press enter to continue\n")
    for itr in xrange(N_iters):
        print "SVI Iter: ", itr, "\tStepsize: ", stepsize[itr]
        test_model.svi_step(stepsize=stepsize[itr])
        # plls.append(test_model.heldout_log_likelihood(S_test, F=F_test))
        samples.append(test_model.copy_sample())

        # print ""
        # print "VB iteration ", itr
        # print "VLB:         ", vlbs[-1]

        # Update plot
        if itr % 1 == 0:
            update_plots(itr, test_model, S, lns, im_net)
    plt.ioff()

    ###########################################################
    # Analyze the samples
    ###########################################################
    analyze_samples(true_model, None, samples)

def initialize_plots(true_model, test_model, S):
    N = true_model.N
    C = true_model.network.C
    data = test_model.data_list[0]
    T = data["T"]
    true_model.add_data(data["S"])
    R = true_model.compute_rate(test_model.data_list[0])
    # Plot the true network
    plt.ion()
    plt.imshow(true_model.weight_model.W_effective.sum(2), vmax=1.0, vmin=-1.0, interpolation="none", cmap="RdGy")
    plt.pause(0.001)


    # Plot the true and inferred firing rate
    plt.ion()
    lns = []
    plt.figure(2)
    for n in xrange(N):
        plt.subplot(N,1,n+1)
        plt.plot(np.arange(T), R[:,n], '-k', lw=2)
        data = test_model.data_list[0]
        lns.append(plt.plot(np.arange(T), test_model.mf_expected_rate(data)[:,n], '-r')[0])
        plt.ylim([0,1])
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
    im_net = plt.imshow(test_model.weight_model.mf_expected_W().sum(2), vmax=1.0, vmin=-1.0, interpolation="none", cmap="RdGy")
    plt.colorbar()
    plt.pause(0.001)

    plt.show()
    plt.pause(0.001)

    # return ln, im_net, im_clus
    return lns, im_net

def update_plots(itr, test_model, S, lns, im_net):
    N = test_model.N
    C = test_model.network.C
    data = test_model.data_list[0]
    T = data["T"]
    plt.figure(2)
    for n in xrange(N):
        plt.subplot(N,1,n+1)
        lns[n].set_data(np.arange(T), test_model.mf_expected_rate(data)[:,n])
    plt.title("Iteration %d" % (itr))
    plt.pause(0.001)

    # plt.figure(3)
    # KC = np.zeros((K,C))
    # KC[np.arange(K), test_model.network.c] = 1.0
    # im_clus.set_data(KC)
    # plt.title("KxC: Iteration %d" % itr)
    # plt.pause(0.001)

    plt.figure(4)
    plt.title("W: Iteration %d" % itr)
    im_net.set_data(test_model.weight_model.mf_expected_W().sum(2))
    plt.pause(0.001)

def analyze_samples(true_model, init_model, samples):
    N_samples = len(samples)
    # Compute sample statistics for second half of samples
    A_samples = np.array([s.weight_model.A     for s in samples])
    W_samples = np.array([s.weight_model.W     for s in samples])
    b_samples = np.array([s.bias_model.b       for s in samples])
    c_samples = np.array([s.network.c          for s in samples])
    p_samples = np.array([s.network.p          for s in samples])
    # mu_samples = np.array([s.network.v          for s in samples])
    # vlbs      = np.array(vlbs)

    offset = N_samples // 2
    A_mean = A_samples[offset:, ...].mean(axis=0)
    W_mean = W_samples[offset:, ...].mean(axis=0)
    b_mean = b_samples[offset:, ...].mean(axis=0)
    p_mean = p_samples[offset:, ...].mean(axis=0)


    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "b true:        ", true_model.bias_model.b
    print ""
    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "b mean:        ", b_mean
    print "p mean:        ", p_mean

    # plt.figure()
    # plt.plot(np.arange(N_samples), vlbs, 'k')
    # plt.xlabel("Iteration")
    # plt.ylabel("VLB")
    # plt.show()

    # # Predictive log likelihood
    # pll_init = init_model.heldout_log_likelihood(S_test)
    # plt.figure()
    # plt.plot(np.arange(N_samples), pll_init * np.ones(N_samples), 'k')
    # plt.plot(np.arange(N_samples), plls, 'r')
    # plt.xlabel("Iteration")
    # plt.ylabel("Predictive log probability")
    # plt.show()

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


    # # Compute the adjusted mutual info score of the clusterings
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
    #
    #
    # plt.ioff()
    # plt.show()


demo(1234)
