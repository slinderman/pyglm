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
    test_model.add_data(train)

    # Initialize the test model parameters with the
    # parameters of the L1-regularized logistic regression model
    test_model.initialize_with_standard_model(init_model)

    # Convolve the test data for fast heldout likelihood calculations
    F_test = test_model.basis.convolve_with_basis(test)


    ###########################################################
    # Fit the test model with batch variational inference
    ###########################################################
    N_iters = 1000
    samples = [test_model.copy_sample()]
    lps = [test_model.log_probability()]
    plls = [test_model.heldout_log_likelihood(test, F=F_test)]

    timestamps = [0]
    start = time.clock()
    for itr in xrange(N_iters):
        lps.append(test_model.log_probability())
        plls.append(test_model.heldout_log_likelihood(test, F=F_test))
        samples.append(test_model.copy_sample())
        timestamps.append(time.clock()-start)

        print ""
        print "Gibbs iteration ", itr
        print "LP:  ", lps[-1]
        print "PLL: ", plls[-1]

        test_model.resample_model()


        # Save intermediate sample
        with gzip.open(
                os.path.join(res_dir,
                             "gibbs.itr%04d.pkl.gz" % itr),
                "w") as f:
            cPickle.dump((test_model.copy_sample(), timestamps[-1]), f, protocol=-1)

    ###########################################################
    # Save the results
    ###########################################################
    results_path = os.path.join(res_dir, "gibbs.pkl.gz")
    with gzip.open(results_path, 'w') as f:
        cPickle.dump((samples, lps, plls, timestamps), f, protocol=-1)


demo(seed=1234)
