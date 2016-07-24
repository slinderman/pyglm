import numpy as np
import os
import cPickle
import gzip

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.switching_models import SwitchingPopulation

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
    base_path = os.path.join("data", "synthetic", "synthetic_K2_C1_T10000")
    data_path = base_path + ".pkl.gz"
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)

    test_data_path = base_path + "_test.pkl.gz"
    with gzip.open(test_data_path, 'r') as f:
        S_test, _ = cPickle.load(f)

    M      = 2                 # Set the max number of states
    T      = S.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    # Set the HDP-HMM hyperparameters
    hdp_hmm_hypers={"alpha": 3.0, "gamma": 4.0,
                    "init_state_concentration": 1.}

    # Copy the network hypers.
    test_model = SwitchingPopulation(N=N, M=M, dt=dt, dt_max=dt_max, B=B,
                                     basis_hypers=true_model.basis_hypers,
                                     observation_hypers=true_model.observation_hypers,
                                     activation_hypers=true_model.activation_hypers,
                                     weight_hypers=true_model.weight_hypers,
                                     network_hypers=true_model.network_hypers,
                                     hdp_hmm_hypers=hdp_hmm_hypers)
    test_model.add_data(S)

    ###########################################################
    # Initialize online plots
    ###########################################################
    plt.ion()
    plt.figure()
    ss_ln = plt.plot(test_model.hidden_state_sequence[0])[0]
    plt.xlabel("Time")
    plt.ylabel("Latent state")
    plt.show()

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 100
    samples = []
    lls = []
    # plls = []
    for itr in xrange(N_samples):
        lls.append(test_model.log_likelihood())
        samples.append(test_model.copy_sample())

        print ""
        print "Gibbs iteration ", itr
        print "LL: ", lls[-1]

        test_model.resample_model()

        # # Update plot
        # if itr % 1 == 0:
        #     ss_ln.set_data(np.arange(T), test_model.hidden_state_sequence[0])
        #     plt.pause(0.0001)
    plt.ioff()

    ###########################################################
    # Plot the log likelihood as a function of iteration
    # Also plot the latent state sequence
    ###########################################################
    plt.figure()
    plt.plot(lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")
    plt.show()

demo()