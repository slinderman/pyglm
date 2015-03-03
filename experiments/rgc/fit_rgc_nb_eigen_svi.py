import numpy as np
import os
import cPickle
import gzip
import time

from pyglm.models import NegativeBinomialEigenmodelPopulation
from pyglm.utils.experiment_helper import load_data, load_results

def demo(dataset="rgc_nb_eigen_300T", run=1, seed=None):
    """
    Fit a weakly sparse
    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    train, test, _ = load_data(dataset)
    train = train.astype(np.int32)
    test = test.astype(np.int32)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    assert os.path.exists(res_dir), "Results directory does not exist: " + res_dir
    standard_results = load_results(dataset, run=run,
                                    algorithms=["bfgs"])

    T      = train.shape[0]
    N      = train.shape[1]
    B      = 5
    dt     = 1.0
    dt_max = 10.0

    # Create and fit a standard model for initialization
    init_model = standard_results["bfgs"]

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################
    # Use the initial model to set hypers
    observation_hypers = {"xi": init_model.xi}
    bias_hypers = {"mu_0": init_model.bias.mean(),
                   "sigma_0": init_model.bias.var()}

    network_hypers = {"mu_0": init_model.W.mean(axis=(0,1)),
                      "Sigma_0": np.diag(init_model.W.var(axis=(0,1)))}

    # Copy the network hypers.
    test_model = NegativeBinomialEigenmodelPopulation(
        N=N, dt=dt, dt_max=dt_max, B=B,
        basis_hypers=init_model.basis_hypers,
        observation_hypers=observation_hypers,
        bias_hypers=bias_hypers,
        network_hypers=network_hypers)

    # Add the data in minibatches
    test_model.add_data(train, minibatchsize=1000)

    # Initialize the test model parameters with the
    # parameters of the L1-regularized model
    test_model.initialize_with_standard_model(init_model)

    # Convolve the test data for fast heldout likelihood calculations
    F_test = test_model.basis.convolve_with_basis(test)


    ###########################################################
    # Fit the test model with stochastic variational inference
    ###########################################################
    N_samples = 1000
    delay = 1.0
    forgetting_rate = 0.25
    stepsize = (np.arange(N_samples) + delay)**(-forgetting_rate)


    samples = [test_model.copy_sample()]
    vlbs = []
    plls = [test_model.heldout_log_likelihood(test, F=F_test)]
    timestamps = [0]
    start = time.clock()
    for itr in xrange(N_samples):
        print ""
        print "SVI iteration ", itr, ".\tStep size: %.3f" % stepsize[itr]
        # print "VLB: ", vlbs[-1]
        print "Pred LL:      ", plls[-1]

        test_model.svi_step(stepsize=stepsize[itr])
        # vlbs.append(test_model.get_vlb())

        # Resample from MF
        # Compute pred ll for variational mode (mean for Gaussian)
        test_model.resample_from_mf()
        test_model.weight_model.mf_mode()
        test_model.bias_model.mf_mode()

        plls.append(test_model.heldout_log_likelihood(test, F=F_test))
        samples.append(test_model.copy_sample())
        timestamps.append(time.clock()-start)

        # Save intermediate sample
        with gzip.open(
                os.path.join(res_dir,
                             "svi.itr%04d.pkl.gz" % itr),
                "w") as f:
            cPickle.dump((test_model.copy_sample(), timestamps[-1]), f, protocol=-1)

    ###########################################################
    # Save the results
    ###########################################################
    results_path = os.path.join(res_dir, "svi.pkl.gz")
    with gzip.open(results_path, 'w') as f:
        cPickle.dump((samples, vlbs, plls, timestamps), f, protocol=-1)


demo(seed=1234)
