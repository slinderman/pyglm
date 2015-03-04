import numpy as np
import os
import sys
import cPickle
import gzip

from pyglm.models import StandardNegativeBinomialPopulation
from pyglm.utils.experiment_helper import load_data

def fit_with_bfgs(dataset, run, seed=None):
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
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    assert os.path.exists(res_dir), "Results directory does not exist: " + res_dir


    T      = train.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    test_model = StandardNegativeBinomialPopulation(N=N, xi=10, dt=dt, dt_max=dt_max, B=B,
                                                    basis_hypers=true_model.basis_hypers)
    test_model.add_data(train)

    ###########################################################
    # Fit the test model with L1-regularized logistic regression
    ###########################################################
    test_model.fit(L1=True)

    ###########################################################
    # Save the fit model
    ###########################################################
    results_path = os.path.join(res_dir, "bfgs.pkl.gz")
    with gzip.open(results_path, 'w') as f:
        cPickle.dump(test_model, f, protocol=-1)


args = sys.argv
assert len(args) == 3
dataset = args[1]
run = int(args[2])

print "Dataset: ", dataset
print "Run:     ", run

fit_with_bfgs(dataset, run, seed=11223344)
