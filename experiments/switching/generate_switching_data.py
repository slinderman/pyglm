import cPickle
import os
import gzip
import numpy as np

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

from hips.plotting.layout import create_figure, create_axis_at_location
from hips.plotting.colormaps import harvard_colors, gradient_cmap

colors = harvard_colors()
blrdcmap = gradient_cmap([colors[1], np.ones(3), colors[0]])

# allcmap = gradient_cmap(harvard_colors())
# cm.register_cmap("harvard", allcmap)
# plt.set_cmap("harvard")

from pyglm.switching_models import NegativeBinomialHDPHSMM

def generate_synthetic_data(seed=None):
    """
    Create a population model and generate a spike train from it.
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    ###########################################################
    # Create a population model
    ###########################################################
    N = 20          # Number of neurons
    M = 5          # Number of states
    T = 10000       # Number of time bins
    T_test = 10000
    dt = 1.0        # Time bin width
    dt_max = 10.0   # Max time of synaptic influence
    B = 1           # Number of basis functions for the weights

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -5.0, "sigma_0": 1.0}

    ###########################################################
    #   Network hyperparameters
    ###########################################################
    network_hypers = {"p": 0.1, "mu_0": np.zeros(B), "Sigma_0": 0.3**2*np.eye(B)}

    hdp_hmm_hypers = {"alpha": 6., "gamma": 6.,
                      "init_state_concentration": 6.}

    duration_hypers = {'alpha_0':2*80, 'beta_0':2}

    ###########################################################
    # Create the model with these parameters
    ###########################################################
    true_model = \
        NegativeBinomialHDPHSMM(
            N=N, M=M, dt=dt, dt_max=dt_max, B=B,
            bias_hypers=bias_hypers,
            network_hypers=network_hypers,
            hdp_hmm_hypers=hdp_hmm_hypers,
            duration_hypers=duration_hypers)

    ###########################################################
    # Sample from the true model
    ###########################################################
    S, stateseq = true_model.generate(T=T, keep=True, keep_stateseq=True)
    print "Number of generated spikes:\t", S.sum(0)

    # Sample test data
    S_test, _ = true_model.generate(T=T_test, keep=False)

    ###########################################################
    #  Plot the network, the spike train and mean rate
    ##########################################################
    pad = 0.5
    sz = 2
    width = 2* pad + M * (sz+pad)
    # fig = create_figure(figsize=(2* pad + M * (sz+pad), 2*pad + sz))
    # for m in xrange(M):
    #     population = true_model.populations[m]
    #     # ax = fig.add_subplot(1,M,m+1)
    #     ax = create_axis_at_location(fig, pad + m * (sz + pad),
    #                                  pad,
    #                                  sz, sz)
    #     im = ax.imshow(np.kron(population.weight_model.W_effective.sum(2),
    #                       np.ones((10,10))),
    #                vmin=-1.0, vmax=1.0,
    #                interpolation="none", cmap=blrdcmap,
    #                extent=(1,N,N,1))
    #     # plt.colorbar(im, ticks=[-1,0,1])
    #
    #     if m == 0:
    #         ax.set_ylabel("$n'$")
    #     ax.set_xlabel("$n$")
    #     ax.set_title("Network %d" % (m+1))
    #
    # # Add colorbar
    # cbax = create_axis_at_location(fig,  M * (sz+pad) + 0.2, pad, 0.2, sz)
    # cb = plt.colorbar(im, cax=cbax)
    # cb.set_ticks([-1, -0.5, 0, 0.5, 1.0])
    # cb.set_ticklabels(['-1', '-0.5', '0', '+0.5', '+1.0'])
    #
    # # plt.tight_layout()
    # fig.savefig("latent_networks.pdf")

    fig = create_figure(figsize=(width,3.5))
    true_model.plot(fig=fig, plot_slice=slice(0,2000))

    fig.savefig("switching_synthetic_data.pdf")

    plt.show()


    # ###########################################################
    # # Pickle and save the data
    # ###########################################################
    # out_dir  = os.path.join('data', "switching_N20_M10_T10000")
    # train_name = "train.pkl.gz"
    # train_path = os.path.join(out_dir, train_name)
    # with gzip.open(train_path, 'w') as f:
    #     print "Saving output to ", train_path
    #     cPickle.dump(S, f, protocol=-1)
    #
    # model_name = "model.pkl.gz"
    # model_path = os.path.join(out_dir, model_name)
    # with gzip.open(model_path, 'w') as f:
    #     print "Saving output to ", model_path
    #     cPickle.dump(true_model.copy_sample(), f, protocol=-1)
    #
    # # Pickle and save the data
    # test_name = "test.pkl.gz"
    # test_path = os.path.join(out_dir, test_name)
    # with gzip.open(test_path, 'w') as f:
    #     print "Saving output to ", test_path
    #     cPickle.dump(S_test, f, protocol=-1)

generate_synthetic_data(12341234)