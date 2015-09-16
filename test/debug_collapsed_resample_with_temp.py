# Debug resampling with temperature

import numpy as np
from pyglm.utils.utils import logistic
from pybasicbayes.util.text import progprint_xrange
import pypolyagamma as ppg

import matplotlib.pyplot as plt

from scipy.stats import norm, probplot

# Use a simple Normal-Bernoulli model
# z ~ N(z | 0, 1)
# x ~ [Bern(x | \sigma(z))]^{1/T} = Bern(x | \sigma(z / T))
# Where T is the temperature of the tempered distribution in [1, \inf)
# When T=1 we target the posterior. When T=\inf we target the prior
T = 0.25
mu_z = 0.0
sigma_z = 1.0

# Initialize Polya-gamma samplers
num_threads = ppg.get_omp_num_threads()
seeds = np.random.randint(2**16, size=num_threads)
ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

def kappa(x):
    # Compute kappa = [a(x) - b(x)/2.] / T
    # for the Bernoulli model where a(x) = x and b(x) = 1
    return (x - 0.5) / T

def resample_z(x, omega):
    # Resample z from its Gaussian conditional
    prior_J = 1./sigma_z
    prior_h = prior_J * mu_z

    lkhd_J = omega
    lkhd_h = kappa(x)

    # Compute the posterior parameters of W
    post_J = prior_J + lkhd_J
    post_mu = (prior_h + lkhd_h) / post_J

    z_smpl = post_mu + np.sqrt(1./post_J) * np.random.randn()
    return z_smpl

def resample_x(z):
    # Resample x from its (scaled) Bernoulli lkhd
    p = logistic(z / T)
    x_smpl = np.random.rand() < p
    return x_smpl

def resample_omega(z, x):
    # Resample with Jesse Windle's ported code
    b = 1. / T
    omega = np.zeros(1)
    psi = z
    ppg.pgdrawvpar(ppgs, np.array([b]), np.array([psi]), omega)

    return omega[0]

def resample_sweep(x,z):
    # Sample an auxiliary variable
    omega = resample_omega(z,x)
    z = resample_z(x, omega)

    # Sample new data
    x = resample_x(z)

    return x,z

def geweke_test(N_samples=10000):
    # Sample from the prior
    z0 = mu_z + np.sqrt(sigma_z) * np.random.randn()
    x0 = np.random.rand() < logistic(z0 / T)

    # Collect samples from the joint
    xs = [x0]
    zs = [z0]

    for smpl in progprint_xrange(1,N_samples):
        x,z = resample_sweep(xs[smpl-1], zs[smpl-1])
        xs.append(x)
        zs.append(z)

    # Make Q-Q plots of the samples
    fig = plt.figure()
    z_ax = fig.add_subplot(121)
    z_dist = norm(mu_z, np.sqrt(sigma_z))
    probplot(np.array(zs), dist=z_dist, plot=z_ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(zs, 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, z_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

geweke_test()




