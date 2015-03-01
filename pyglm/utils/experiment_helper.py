import os
import gzip
import cPickle

"""
Helper functions for organizing experimental results
"""

def load_data(dataset=""):
    base_dir = os.path.join("data", dataset)
    assert os.path.exists(base_dir), \
        "Could not find data directory: " + base_dir

    model_path = os.path.join(base_dir, "model.pkl.gz")
    model = None
    if os.path.exists(model_path):
        with gzip.open(model_path, "r") as f:
            model = cPickle.load(f)

    train_path = os.path.join(base_dir, "train.pkl.gz")
    with gzip.open(train_path, "r") as f:
        train = cPickle.load(f)

    test_path = os.path.join(base_dir, "test.pkl.gz")
    with gzip.open(test_path, "r") as f:
        test = cPickle.load(f)

    return train, test, model

def load_results(dataset="", run=0,
                 algorithms=("bfgs", "gibbs","vb")):

    base_dir = os.path.join("results", dataset, "run%03d" % run)
    assert os.path.exists(base_dir), \
        "Could not find results directory: " + base_dir

    results = {}
    for alg in algorithms:
        res_path = os.path.join(base_dir, alg + ".pkl.gz")
        if os.path.exists(res_path):
            print "\tLoading ", alg, " results..."
            with gzip.open(res_path, "r") as f:
                results[alg] = cPickle.load(f)

    return results