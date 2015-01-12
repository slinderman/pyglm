from __future__ import division
import os
import cPickle

import numpy as np
import matplotlib.pyplot as plt

from pyglm.populations import ErdosRenyiNegativeBinomialPopulation
from deps.pybasicbayes.distributions import DiagonalGaussian
from pyglm.utils.basis import  Basis


np.random.seed(0)

#
#  Load data
#
dat_file = os.path.join('data', 'wilson.pkl')
with open(dat_file) as f:
    raw_data = cPickle.load(f)

S = raw_data['S']
T,N = S.shape
dt = raw_data['dt']

################
#  parameters  #
################
res_file = os.path.join('results', 'hippocampus_er.pkl')

N_states = 4
alpha = 3.
gamma = 3.
N_samples = 1000

# Basis parameters
B = 3       # Number of basis functions
dt_max = 1.0      # Number of time bins over which the basis extends
basis_parameters = {'type' : 'cosine',
                    'n_eye' : 0,
                    'n_bas' : B,
                    'a' : 1.0/120,
                    'b' : 0.5,
                    'L' : 100,
                    'orth' : False,
                    'norm' : True
                    }
basis = Basis(B, dt, dt_max, basis_parameters)

#############################
#  generate synthetic data  #
#############################
spike_train_hypers = {'xi' : 10,
                      'n_iters_per_resample' : 1}

global_bias_hypers= {'mu_0' : -5.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 0.1,
                     'nu_0' : 1.0
                    }

network_hypers = {'rho' : 0.1,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0,
                          'alphas_0' : 1.0,
                          'betas_0' : 1.0,
                      },
                  'refractory_rho' : 0.1,
                  'refractory_prior_class' : DiagonalGaussian,
                  'refractory_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0,
                          'alphas_0' : 1.,
                          'betas_0' : 1.
                      },
                 }

print "Spike counts: "
print S.sum(0)
print ""

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
# population.initialize_to_empty()

#
# plotting  #
#
neurons_to_plot = [0,1,2,16,17,18]
t_lim = [50,60]
axs, lns = population.plot_mean_spike_counts(np.hstack(fS), dt=dt,
                                             S=S, color='k',
                                             inds=neurons_to_plot,
                                             t_lim=t_lim)
net_axs, net_lns = population.plot_weighted_network()

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
    if i % 10 == 0:
        with open(res_file + ".%i" % i, 'w') as f:
            cPickle.dump((A_smpls, W_smpls, bias_smpls, sigma_smpls, population), f, protocol=-1)


# END Profiling
pr.disable()

s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()

# Save the results
print "Saving results"
with open(res_file, 'w') as f:
    cPickle.dump((A_smpls, W_smpls, bias_smpls, sigma_smpls), f)

