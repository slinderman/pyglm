import numpy as np
import os
import cPickle
import gzip
# np.seterr(all='raise')

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.models import NegativeBinomialPopulation
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
    run = 1
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
    test_model = NegativeBinomialPopulation(N=N, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            bias_hypers=true_model.bias_hypers,
                            weight_hypers=true_model.weight_hypers,
                            network_hypers={"p": 0.19})

    # Add the data in minibatches
    test_model.add_data(train, minibatchsize=1000)
    test_model.initialize_with_standard_model(init_model)
    F_test = test_model.basis.convolve_with_basis(test)

    # Initialize plots
    im_net = initialize_plots(true_model, test_model, train)

    ###########################################################
    # Fit the test model with stochastic variational inference (SVI)
    ###########################################################
    # Stochastic variational inference
    N_iters = 50
    samples = []
    plls = []
    delay = 1.0
    forgetting_rate = 0.5
    stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
    raw_input("Press enter to continue\n")
    for itr in xrange(N_iters):
        print "SVI Iter: ", itr, "\tStepsize: ", stepsize[itr]
        test_model.svi_step(stepsize=stepsize[itr])
        plls.append(test_model.heldout_log_likelihood(test, F=F_test))
        samples.append(test_model.copy_sample())

        # print ""
        # print "VB iteration ", itr
        print "PLL:     ", plls[-1]

        # Update plot
        if itr % 1 == 0:
            update_plots(itr, test_model, im_net)
    plt.ioff()

    ###########################################################
    # Analyze the samples
    ###########################################################
    analyze_samples(true_model, init_model, samples, test, plls)

def initialize_plots(true_model, test_model, S):
    data = test_model.data_list[0]
    true_model.add_data(data["S"])
    # Plot the true network
    plt.ion()
    plt.imshow(true_model.weight_model.W_effective.sum(2), vmax=1.0, vmin=-1.0, interpolation="none", cmap="RdGy")
    plt.pause(0.001)

    plt.figure(4)
    im_net = plt.imshow(test_model.weight_model.mf_expected_W().sum(2), vmax=1.0, vmin=-1.0, interpolation="none", cmap="RdGy")
    plt.colorbar()
    plt.pause(0.001)

    plt.show()
    plt.pause(0.001)
    return im_net

def update_plots(itr, test_model, im_net):
    plt.figure(4)
    plt.title("W: Iteration %d" % itr)
    im_net.set_data(test_model.weight_model.mf_expected_W().sum(2))
    plt.pause(0.001)

def analyze_samples(true_model, init_model, samples, test, plls):
    N_samples = len(samples)

    # Predictive log likelihood
    pll_init = init_model.heldout_log_likelihood(test)
    plt.figure()
    plt.plot(np.arange(N_samples), pll_init * np.ones(N_samples), 'k')
    plt.plot(np.arange(N_samples), plls, 'r')
    plt.xlabel("Iteration")
    plt.ylabel("Predictive log probability")
    plt.show()


demo(11223344)
