from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from pyglm.populations import ErdosRenyiNegativeBinomialPopulation, \
                        ErdosRenyiBernoulliPopulation
from pyglm.deps.pybasicbayes.distributions import DiagonalGaussian
from pyglm.utils.basis import  Basis


seed = np.random.randint(2**16)
# seed = 1234
print "Setting random seed to ", seed
np.random.seed(seed)

################
#  parameters  #
################
N = 1
dt = 0.001
T = 10000
N_samples = 1000

# Basis parameters
B = 2       # Number of basis functions
dt_max = 0.1      # Number of time bins over which the basis extends
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

#############################
#  generate synthetic data  #
#############################
observation = 'bernoulli'
spike_train_hypers = {}

# global_bias_class = GaussianFixed
# global_bias_hypers= {'mu' : -3,
#                      'sigma' : 0.001}
global_bias_hypers = {
                     'mu_0' : -3.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 1.0,
                     'nu_0' : 1.0
                    }

network_hypers = {'rho' : 0.5,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N**2,
                          'alphas_0' : 10.0,
                          'betas_0' : 10.0
                      },
                  'refractory_rho' : 0.5,
                  'refractory_prior_class' : DiagonalGaussian,
                  'refractory_prior_hypers' :
                      {
                          'mu_0' : -3.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N,
                          'alphas_0' : 10.,
                          'betas_0' : 10.
                      },
                 }
population = ErdosRenyiBernoulliPopulation(
        N, basis,
        global_bias_hypers=global_bias_hypers,
        neuron_hypers=spike_train_hypers,
        network_hypers=network_hypers,
        )

inf_population = ErdosRenyiBernoulliPopulation(
        N, basis,
        global_bias_hypers=global_bias_hypers,
        neuron_hypers=spike_train_hypers,
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
raw_input("Press any key to continue")
#
# Debug
#
true_bias = population.biases.copy()
# inf_population = population
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
inf_population.add_data(data)
inf_population.initialize_to_empty()


ll_samples = []
A_samples = []
w_samples = []
bias_samples = []

for s in range(N_samples):
    print "Iteration ", s
    ll = inf_population.heldout_log_likelihood(data)
    print "LL: ", ll
    inf_population.resample_model()

    # Plot this sample
    inf_population.plot_mean_spike_counts(Xs, dt=dt, lns=inf_lns)

    # Collect samples
    ll_samples.append(ll)
    A_samples.append(inf_population.A.copy())
    bias_samples.append(inf_population.biases.copy())

    plt.pause(0.01)

    # DEBUG
    # print inf_population.network.refractory_prior
    # print inf_population.spike_train_models[0].model.regression_models[0].weights_prior

A_mean = np.array(A_samples)[N_samples/2:].mean(0)
print "True A: \n", population.A
print "Mean A: \n", A_mean

bias_mean = np.array(bias_samples)[N_samples/2:].mean(0)
print "True bias: ", population.biases
print "Mean bias: ", bias_mean

plt.figure()
plt.plot(np.array(ll_samples))
plt.show()

# for itr in progprint_xrange(25,perline=5):
#     model.resample_model()

