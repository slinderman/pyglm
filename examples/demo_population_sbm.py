from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from oldpyglm.populations import SBMBernoulliPopulation
from oldpyglm.utils.basis import  Basis


seed = np.random.randint(2**16)
print "Setting random seed to ", seed
np.random.seed(seed)

################
#  parameters  #
################
N = 6
K = 2           # Number of latent classes
dt = 0.001
T = 1000
N_samples = 1000

# Basis parameters
B = 2              # Number of impulse response basis functions
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
K = 2               # Number of latent classes
latent_variable_hypers = {'K' : K,
                          'alpha' : 1.0,
                          'gamma' : 10.0
                         }

spike_train_hypers = {}
global_bias_hypers= {'mu' : -4,
                     'sigmasq' : 1.0,
                     'mu_0' : -5.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 0.1,
                     'nu_0' : 10.0
                    }
network_hypers = {'rho' : 0.5,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          # 'nus_0' : 1.0,
                          # 'alphas_0' : 1.0,
                          # 'betas_0' : 0.001,
                          'lmbda_0' : np.eye(basis.B)
                      },
                  'refractory_rho' : 0.9,
                  'refractory_prior_hypers' :
                      {
                          'mu_0' : -3.0 * np.ones((basis.B,)),
                          # 'nus_0' : 1.0,
                          # 'alphas_0' : 1.,
                          # 'betas_0' : 0.001
                          'lmbda_0' : np.eye(basis.B)
                      },
                 }

population = SBMBernoulliPopulation(
        N, basis,
        latent_variable_hypers=latent_variable_hypers,
        global_bias_hypers=global_bias_hypers,
        neuron_hypers=spike_train_hypers,
        network_hypers=network_hypers,
        )

inf_population = SBMBernoulliPopulation(
        N, basis,
        latent_variable_hypers=latent_variable_hypers,
        global_bias_hypers=global_bias_hypers,
        neuron_hypers=spike_train_hypers,
        network_hypers=network_hypers,
        )

S, X = population.rvs(size=T, return_X=True)
X = X[:T,:]
data = np.hstack((X,S))

print "A: ",
print population.A
print ""
print "biases: ",
print population.biases
print ""

print "Spike counts: "
print S.sum(0)
print ""

#############
# plotting  #
#############
axs, true_lns = population.plot_mean_spike_counts(X, dt=dt, S=S, color='k')
_, inf_lns = inf_population.plot_mean_spike_counts(X, axs=axs, dt=dt, color='r')
net_axs, net_lns = population.plot_weighted_network()
inf_net_axs, inf_net_lns = inf_population.plot_weighted_network()
sbm_axs, sbm_lns = population.plot_sbm_weights(color='k')
inf_sbm_axs, inf_sbm_lns = inf_population.plot_sbm_weights(color='r', axs=sbm_axs)

plt.ion()
plt.show()


#############
#  sample!  #
#############
for s in range(N_samples):
    print "Iteration ", s
    inf_population.resample(data)


    # Plot this sample
    inf_population.plot_mean_spike_counts(X, dt=dt, lns=inf_lns)
    inf_population.plot_weighted_network(lns=inf_net_lns)
    inf_population.plot_sbm_weights(lns=inf_sbm_lns)

    # import pdb; pdb.set_trace()
    for fig in range(1,5):
        plt.figure(fig)
        plt.pause(0.1)

# for itr in progprint_xrange(25,perline=5):
#     model.resample_model()

