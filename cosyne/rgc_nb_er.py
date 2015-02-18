import os
import cPickle
import gzip

import numpy as np
import matplotlib.pyplot as plt

from deps.pybasicbayes.distributions import DiagonalGaussian
from oldpyglm.utils.datahelper import load_data
from utils.basis import  Basis
from oldpyglm.populations import ErdosRenyiNegativeBinomialPopulation





# seed = np.random.randint(2**16)
seed = 0
print "Seed: ", seed
np.random.seed(seed)

#
#  Load data
#
dat_file = os.path.join('data', 'rgc.mat')
data = load_data(dat_file)

# Bin with larger bins
dt_orig = 0.001
dt = 0.01
S_orig = data['S']
T,N = S_orig.shape
S = S_orig.reshape((-1, int(dt/dt_orig), N)).sum(1)
T,N = S.shape
print "Spike counts: "
print S_orig.sum(0)
print S.sum(0)

#
#  Parameters
#
plot = False
init_file = os.path.join('results', 'rgc_logistic_regression.pkl')
res_dir = os.path.join('results', 'rgc_nb_er', 'runs', '2')
N_samples = 500

# Basis parameters
B = 3              # Number of impulse response basis functions
dt_max = 0.05      # Duration of impulse response
basis_parameters = {'type' : 'cosine',
                    'n_eye' : 0,
                    'n_bas' : B,
                    'a' : 1.0/120,
                    'b' : 0.5,
                    'L' : 100,
                    'orth' : False,
                    'norm' : False
                    }
basis = Basis(B, dt, dt_max, basis_parameters)

#
# Initialize the model
#
spike_train_hypers = {'xi' : 10}
global_bias_hypers= {'mu' : -4,
                     'sigmasq' : 1.0,
                     'mu_0' : -5.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 0.1,
                     'nu_0' : 10.0
                    }
network_hypers = {'rho' : 0.1,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N**2,
                          'alphas_0' : 1.0,
                          'betas_0' : 1.0
                      },
                  'refractory_rho' : 0.9,
                  'refractory_prior_class' : DiagonalGaussian,
                  'refractory_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N,
                          'alphas_0' : 1.,
                          'betas_0' : 1.
                      },
                 }

population = ErdosRenyiNegativeBinomialPopulation(
    N, basis,
    global_bias_hypers=global_bias_hypers,
    neuron_hypers=spike_train_hypers,
    network_hypers=network_hypers,
    )

# Filter the spike train
fS = population.filter_spike_train(S)
full_data = np.hstack((np.hstack(fS), S))

# Add the data to the population
# NOTE that we are treating the population like a model in that
# we are assuming this data is static
population.add_data(full_data)
# Initialize the parameters with an empty network
population.initialize_to_empty()

# if os.path.exists(init_file):
#     with open(init_file, 'r') as f:
#         A0, weights0, biases0 = cPickle.load(f)
#         population.A = A0
#         population.weights = weights0
#         population.biases = biases0
# else:
#     population.initialize_with_logistic_regression()
#     A0,weights0,biases0 = population.A, population.weights, population.biases
#     with open(init_file, 'w') as f:
#         cPickle.dump((A0, weights0, biases0), f, protocol=-1)

#
# plotting  #
#
if plot:
    neurons_to_plot = [0,1,2,16,17,18]
    t_lim = [50,60]
    axs, lns = population.plot_mean_spike_counts(np.hstack(fS), dt=dt,
                                                 S=S, color='k',
                                                 inds=neurons_to_plot,
                                                 t_lim=t_lim)
    net_axs, net_lns = population.plot_weighted_network()
    net_axs.plot([0,N+1],[0,N+1], ':k')
    net_axs.plot([0,N+1],[16,16], ':k')
    net_axs.plot([16,16],[0,N+1], ':k')

    plt.ion()
    plt.show()
    plt.pause(0.1)

#
#  Inference
#
A_smpls = []
W_smpls = []
bias_smpls = []
sigma_smpls = []


# Profile the inference
import cProfile, StringIO, pstats
pr = cProfile.Profile()
pr.enable()

for i in range(N_samples):
    print "Iteration ", i
    print "LL: ", population.heldout_log_likelihood(full_data)
    population.resample()

    # Plot this sample
    if plot:
        population.plot_mean_spike_counts(np.hstack(fS), dt=dt, lns=lns, inds=neurons_to_plot)
        population.plot_weighted_network(ax=net_axs, lns=net_lns)

        for fig in range(1,3):
            plt.figure(fig)
            plt.pause(0.1)

        A_smpls.append(population.A)
        W_smpls.append(population.weights)
        bias_smpls.append(population.biases)
        sigma_smpls.append(population.etas)

    # Periodically save results
    # if i % 10 == 0:
    #     with open(res_file + ".%i" % i, 'w') as f:
    #         cPickle.dump((A_smpls, W_smpls, bias_smpls, sigma_smpls), f)

    if i % 10 == 0:
        with gzip.open(os.path.join(res_dir, 'iter%03d' % i),'w') as outfile:
            cPickle.dump(population.parameters,outfile,protocol=-1)



# END Profiling
pr.disable()

s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()

# Save the results
# print "Saving results"
with open(os.path.join(res_dir, 'samples.pkl'), 'w') as f:
    cPickle.dump((A_smpls, W_smpls, bias_smpls, sigma_smpls), f)


