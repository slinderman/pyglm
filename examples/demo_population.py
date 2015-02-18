from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from oldpyglm.populations import ErdosRenyiNegativeBinomialPopulation, \
                        ErdosRenyiBernoulliPopulation
from oldpyglm.deps.pybasicbayes.distributions import DiagonalGaussian
from oldpyglm.utils.basis import  Basis


seed = np.random.randint(2**16)
seed = 62645
print "Setting random seed to ", seed
np.random.seed(seed)

################
#  parameters  #
################
N = 2
dt = 0.001
T = 60000
N_samples = 10000

# Basis parameters
B = 1       # Number of basis functions
dt_max = 0.1

#############################
#  generate synthetic data  #
#############################
observation = 'bernoulli'
if observation == 'negative_binomial':
    neuron_hypers = {'xi' : 10}
else:
    neuron_hypers = {'alpha_0' : 3.0, 'beta_0' : 1.0}

# global_bias_class = GaussianFixed
# global_bias_hypers= {'mu' : -3,
#                      'sigma' : 0.001}
global_bias_hypers = {
                     'mu_0' : -3.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 0.5,
                     'nu_0' : 1.0
                    }

network_hypers = {'rho' : 0.5,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((B,)),
                          'nus_0' : 1.0/N**2,
                          'alphas_0' : 10.0,
                          'betas_0' : 20.0
                      },
                  # 'weight_prior_hypers' :
                  #     {
                  #         'mu' : 0.0 * np.ones((B,)),
                  #         'sigmas' : 1.0/N**2 * np.ones(B)
                  #     },
                  'refractory_rho' : 0.9,
                  'refractory_prior_class' : DiagonalGaussian,
                  'refractory_prior_hypers' :
                      # {
                      #     'mu' : -1.0 * np.ones((B,)),
                      #     'sigmas' : 1.0/N**2 * np.ones(B)
                      # },

                      {
                          'mu_0' : -1.0 * np.ones((B,)),
                          'nus_0' : 1.0/N,
                          'alphas_0' : 10.,
                          'betas_0' : 20.
                      },
                 }
if observation == 'negative_binomial':
    population = ErdosRenyiNegativeBinomialPopulation(
            N, B=B, dt=dt,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=neuron_hypers,
            network_hypers=network_hypers,
            )

    inf_population = ErdosRenyiNegativeBinomialPopulation(
            N, B=B, dt=dt,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=neuron_hypers,
            network_hypers=network_hypers,
            )
else:
    population = ErdosRenyiBernoulliPopulation(
            N, B=B, dt=dt,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=neuron_hypers,
            network_hypers=network_hypers,
            )

    inf_population = ErdosRenyiBernoulliPopulation(
            N, B=B, dt=dt,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=neuron_hypers,
            network_hypers=network_hypers,
            )

S, Xs = population.generate(size=T)
Xs = [X[:T,:] for X in Xs]
data = np.hstack(Xs + [S])

print "A: ",
print population.A
print ""
print "W: ",
print population.weights
print ""
print "biases: ",
print population.biases
print ""

print "Spike counts: "
print S.sum(0)
print ""

##############
# plotting  #
#############
t_lim = [0,1]
axs, true_lns = population.plot_mean_spike_counts(Xs, dt=dt, S=S, color='k', t_lim=t_lim)
_, inf_lns = inf_population.plot_mean_spike_counts(Xs, axs=axs, dt=dt, color='r', style='--')
plt.ion()
plt.show()
plt.pause(0.01)


#############
#  sample!  #
#############
# Initialize the parameters with an empty network
inf_population.add_data(S)
inf_population.initialize_to_empty()

ll_samples = []
A_samples = []
w_samples = []
bias_samples = []

for s in range(N_samples):
    print "Iteration ", s
    ll = inf_population.heldout_log_likelihood(data)
    print "LL: ", ll
    inf_population.resample_model(do_resample_network=False)


    # Collect samples
    ll_samples.append(ll)
    A_samples.append(inf_population.A.copy())
    w_samples.append(inf_population.weights.copy())
    bias_samples.append(inf_population.biases.copy())

    if s % 5 == 0:
        # Plot this sample
        inf_population.plot_mean_spike_counts(Xs, dt=dt, lns=inf_lns)
        plt.pause(0.001)

A_mean = np.array(A_samples)[N_samples/2:,...].mean(0)
print "True A: \n", population.A
print "Mean A: \n", A_mean

w_mean = np.array(w_samples)[N_samples/2:,...].mean(0)
print "True w: \n", population.weights
print "Mean w: \n", w_mean

bias_mean = np.array(bias_samples)[N_samples/2:,...].mean(0)
print "True bias: ", population.biases
print "Mean bias: ", bias_mean

plt.figure()
plt.plot(np.array(ll_samples))
plt.show()

