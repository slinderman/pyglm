import os
import glob
import gzip
import cPickle
import numpy as np
import matplotlib.pyplot as plt

from pyglm.utils.experiment_helper import load_data

samples = []
lps = []
plls = []
timestamps = []

dataset = "rgc_nb_eigen_300T"
run = 1
alg = "gibbs"
_, test, _ = load_data(dataset)
S_test = test.astype(np.int32)

# # Load the test data
# test_path = os.path.join("..", "rgc_test.pkl")
# with open(test_path, 'r') as f:
#     test_data = cPickle.load(f)
#     S_test = test_data["S"].astype(np.int32)

res_dir = os.path.join("results", dataset, "run%3d" % run)
res_files = sorted(glob.glob(alg + ".itr*.pkl.gz"))
for res_file in res_files:
    print "Parsing result ", res_file
    with gzip.open(res_file, "r") as f:
        test_model, timestamp = cPickle.load(f)

    # Compute the log prob and the predictive log likelihood
    lps.append(test_model.log_probability())
    plls.append(test_model.heldout_log_likelihood(S_test))
    samples.append(test_model.copy_sample())
    timestamps.append(timestamp)

print "Saving parsed results"
with gzip.open("rgc_60T.eigen_fit.gibbs.pkl.gz", "w") as f:
    cPickle.dump((samples, lps, plls, timestamps), f, protocol=-1)

