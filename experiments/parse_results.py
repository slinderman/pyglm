import os
import glob
import gzip
import cPickle
import numpy as np
import matplotlib.pyplot as plt

from pyglm.utils.experiment_helper import load_data

samples = []
stateseqs = []
lps = []
plls = []
timestamps = []

dataset = "switching_N20_M10_T10000"
run = 1
alg = "gibbs.empty.hdphsmm"
_, test, _ = load_data(dataset)
S_test = test.astype(np.int32)

# # Load the test data
# test_path = os.path.join("..", "rgc_test.pkl")
# with open(test_path, 'r') as f:
#     test_data = cPickle.load(f)
#     S_test = test_data["S"].astype(np.int32)

res_dir = os.path.join("results", dataset, "run%03d" % run)
res_files = sorted(glob.glob(os.path.join(res_dir, alg + ".itr*.pkl.gz")))
print ""
for res_file in res_files:
    print "Parsing result ", res_file
    with gzip.open(res_file, "r") as f:
        test_model, timestamp = cPickle.load(f)

    # Compute the log prob and the predictive log likelihood
    # lps.append(test_model.log_probability())
    lps.append(0)
    plls.append(test_model.heldout_log_likelihood(S_test))
    # samples.append(test_model.copy_sample())
    stateseqs.append(test_model.hidden_state_sequence[0])
    timestamps.append(timestamp)

print "Saving parsed results"
res_file = os.path.join(res_dir, alg + "-parsed.pkl.gz")
with gzip.open(res_file, "w") as f:
    cPickle.dump((samples, lps, plls, timestamps, stateseqs), f, protocol=-1)

