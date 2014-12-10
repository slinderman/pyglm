from __future__ import division

import numpy as np
from pyhsmm.util.text import progprint_xrange

from deps.pybasicbayes.models import Mixture
from pyglm.populations import ErdosRenyiBernoulliPopulation
from nb.deps.pybasicbayes.distributions import Gaussian
from pyglm.utils.basis import  Basis


np.random.seed(0)

################
#  parameters  #
################

N_states = 4
alpha = 3.
gamma = 3.
N = 2
dt = 0.001
T = 10000
N_samples = 1000

# Basis parameters
B = 3       # Number of basis functions
dt_max = 0.1      # Number of time bins over which the basis extends
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

spike_train_hypers = {}

global_bias_hypers= {'mu' : -4,
                     'sigmasq' : 1.0,
                     'mu_0' : -5.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 0.1,
                     'nu_0' : 100.0
                    }
rho = np.zeros((N,N))
rho[0,1] = 1.0
network_hypers = {'rho' : rho,
                  'weight_prior_class' : Gaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'sigma_0' : 0.0005 * np.eye(basis.B),
                          'nu_0' : basis.B+10,
                          'kappa_0' : .01
                      },
                  'refractory_prior_class' : Gaussian,
                  'refractory_prior_hypers' :
                      {
                          'mu_0' : -1.0 * np.ones((basis.B,)),
                          'sigma_0' : 0.1 * np.eye(basis.B),
                          'nu_0' : basis.B+10,
                          'kappa_0' : .01
                      }
                 }
population = ErdosRenyiBernoulliPopulation(
        N, basis,
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

##################
#  create model  #
##################

obs_distns = \
        [ErdosRenyiBernoulliPopulation(
            N, basis,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=spike_train_hypers,
            network_hypers=network_hypers,
            )
            for _ in xrange(N_states)]
model = Mixture(alpha_0=alpha, components=obs_distns)

model.add_data(data=data)

#############
#  sample!  #
#############

for itr in progprint_xrange(25,perline=5):
    model.resample_model()

