from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.distributions import Regression, Gaussian
from pybasicbayes.util.text import progprint_xrange
from autoregressive.distributions import AutoRegression

from pyglm.utils.utils import logistic

import pyglm.dynamic_models
reload(pyglm.dynamic_models)
from pyglm.dynamic_models import BernoulliLDS, PGEmissions

npr.seed(0)


#########################
#  set some parameters  #
#########################

mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)

C = np.array([[2.,0.]])
D_out, D_in = C.shape


###################
#  generate data  #
###################

truemodel = BernoulliLDS(
    init_dynamics_distn=Gaussian(mu_init, sigma_init),
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    emission_distn=PGEmissions(D_out, D_in, C=C))

T = 2000
data, z_true = truemodel.generate(T)
psi_true = z_true.dot(C.T)
p_true = logistic(psi_true)



###############
#  fit model  #
###############
model = BernoulliLDS(
    init_dynamics_distn=Gaussian(mu_0=np.zeros(D_in), sigma_0=np.eye(D_in),
                                 kappa_0=1.0, nu_0=D_in+1),
    dynamics_distn=AutoRegression(
            sigma=sigma_states, nu_0=D_in+1, S_0=D_in*np.eye(D_in), M_0=np.zeros((D_in, D_in)), K_0=D_in*np.eye(D_in)),
    emission_distn=PGEmissions(D_out, D_in, C=C, sigmasq_C=1.0))
model.add_data(data.X)

N_samples = 1000
def update(model):
    model.resample_model()
    z_inf = model.states_list[0].stateseq
    C_inf = model.C
    psi_inf = z_inf.dot(C_inf.T)
    p_inf = logistic(psi_inf)
    return model.log_likelihood(), p_inf

results = [update(model) for _ in progprint_xrange(N_samples)]
lls = np.array([r[0] for r in results])
ps = np.array([r[1] for r in results])

plt.figure()
plt.plot(data.X, 'bx', ls="none")
# plt.plot(psi, 'r')
plt.plot(p_true, 'r', lw=2)
plt.errorbar(np.arange(T), ps.mean(0), yerr=ps.std(0), fmt='--r', )
plt.ylim(-0.1, 1.1)

plt.figure()
plt.plot(lls)
plt.xlabel("Iteration")
plt.ylabel("LL")
plt.show()