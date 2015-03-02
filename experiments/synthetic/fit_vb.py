import numpy as np
import os
import sys
import cPickle
import gzip
import time

from pyglm.models import NegativeBinomialEigenmodelPopulation
from pyglm.utils.experiment_helper import load_data, load_results

def fit_with_vb(dataset, run, seed=None):
    """
    Fit the dataset using Gibbs sampling
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
    test_model = NegativeBinomialEigenmodelPopulation(N=N, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            weight_hypers=true_model.weight_hypers,
                            bias_hypers=true_model.bias_hypers,
                            network_hypers=true_model.network_hypers)
    test_model.add_data(train)

    # Initialize with the standard model
    test_model.initialize_with_standard_model(init_model)

    # Convolve the test data for fast heldout likelihood calculations
    F_test = test_model.basis.convolve_with_basis(test)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 1000
    samples = [test_model.copy_sample()]
    vlbs = [test_model.get_vlb()]
    plls = [test_model.heldout_log_likelihood(test, F=F_test)]
    timestamps = [0]
    start = time.clock()
    for itr in xrange(N_samples):
        print ""
        print "VB iteration ", itr
        print "VLB: ", vlbs[-1]

        test_model.meanfield_coordinate_descent_step()
        vlbs.append(test_model.get_vlb())

        # Resample from MF
        # Compute pred ll for variational mode (mean for Gaussian)
        test_model.resample_from_mf()
        test_model.weight_model.mf_mode()
        test_model.bias_model.mf_mode()

        plls.append(test_model.heldout_log_likelihood(test, F=F_test))
        samples.append(test_model.copy_sample())
        timestamps.append(time.clock()-start)

    ###########################################################
    # Save the results
    ###########################################################
    results_path = os.path.join(res_dir, "vb.pkl.gz")
    with gzip.open(results_path, 'w') as f:
        cPickle.dump((samples, vlbs, plls, timestamps), f, protocol=-1)



args = sys.argv
assert len(args) == 3
dataset = args[1]
run = int(args[2])

print "Dataset: ", dataset
print "Run:     ", run

fit_with_vb(dataset, run, seed=12341234)
