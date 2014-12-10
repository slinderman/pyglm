from __future__ import division
import os
import cPickle
import gzip

import numpy as np
from pyhsmm.models import WeakLimitHDPHMM
from pyhsmm.util.text import progprint_xrange

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

N_states = 25
alpha = 10.
gamma = 10.
N_samples = 500

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
# NOTE: n_iters_per_resample = 1 probably doesn't work
# but n_iters_per_resample = 10 seems to work!
spike_train_hypers = {'xi' : 10,
                      'n_iters_per_resample' : 15}

global_bias_hypers= {'mu_0' : -3.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 2.0,
                     'nu_0' : 1.0
                    }

network_hypers = {'rho' : 0.0,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0,
                          'alphas_0' : 1.0,
                          'betas_0' : 1.0,
                      },
                  'refractory_rho' : 0.0,
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

# Filter the spike train
fS = basis.convolve_with_basis(S)
full_data = np.hstack((np.hstack(fS), S))

################################
#  initialize with flat model  #
################################

obs_distns = \
        [ErdosRenyiNegativeBinomialPopulation(
            N, basis,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=spike_train_hypers,
            network_hypers=network_hypers,
            )
            for _ in xrange(N_states)]

## initialize
# for o in progprint(obs_distns,perline=1):
#     o.add_data(full_data)
#     for _ in xrange(1):
#         o.resample()
#     for stm in o.spike_train_models:
#         stm.model.data_list = []

##################
#  create model  #
##################

model = WeakLimitHDPHMM(
        alpha=alpha,gamma=gamma,init_state_distn='uniform',
        # alpha_a_0=1.,alpha_b_0=1./10,
        # gamma_a_0=1.,gamma_b_0=1./10,
        obs_distns=obs_distns)

model.add_data(data=full_data)

#############
#  sample!  #
#############

for itr in progprint_xrange(N_samples,perline=1):
    print "Iteration ", itr
    print "Used states: ", np.unique(model._get_used_states())
    print "{:20.4f}".format(model.log_likelihood(full_data))

    with gzip.open('iter%03d' % itr,'w') as outfile:
        cPickle.dump(model,outfile,protocol=-1)

    # for o in model.obs_distns:
    #     print o.biases

    model.resample_model(obs_jobs=4)

