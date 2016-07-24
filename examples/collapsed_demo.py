"""
Hacking together a simple model with 1D weights and a collapsed Gibbs
inference algorithm.
"""
import numpy as np
np.random.seed(1234)

from scipy.misc import logsumexp

import matplotlib.pyplot as plt

from pyglm.utils.basis import CosineBasis
from pyglm.utils.utils import logistic

import pypolyagamma as ppg

# Sample an autoregressive negative binomial model
def sample_spiketrain(T, N, basis, A, W, b):
    L = basis.L
    psi = np.ones((T+L,N)) * b
    S = np.zeros((T+L,N))
    H = basis.basis.ravel()[:,None,None] * W[None,:,:] * A[None,:,:]

    for t in xrange(T):
        # Sample spikes for t-th time bin
        # S[t] =  np.random.negative_binomial(xi, 1.-logistic(psi[t]))
        S[t] =  np.random.rand(N) < logistic(psi[t])

        # Compute change in activation via tensor product
        dpsi = np.tensordot( H, S[t,:], axes=([1, 0]))
        psi[t:t+L,:] += dpsi

    S = S[:T]
    psi = psi[:T]

    return S,psi


def resample_omega():
    ppg.pgdrawvpar(ppgs, np.ones(T*N), psi.ravel(), omega.ravel())

def resample_A():
    # Compute kappa for a bernoulli model
    kappa = S - 0.5

    for n in xrange(N):
        for m in xrange(N):
            # Compute the marginal prob with and without A[m,n]
            lps = np.zeros(2)
            for v in [0,1]:
                A[m,n] = v
                Aeff = np.concatenate(([1], A[:,n])).astype(np.bool)

                # Get effective params for nonzero A's
                F_eff = F_full[:, Aeff]
                mu_eff = mu_full[Aeff]
                Sigma_eff = Sigma_full[np.ix_(Aeff, Aeff)]

                # Compute the posterior parameters of W
                Lambda_post = np.linalg.inv(Sigma_eff) + (F_eff * omega[:,n][:,None]).T.dot(F_eff)
                Sigma_post = np.linalg.inv(Lambda_post)
                mu_post = Sigma_post.dot(np.linalg.solve(Sigma_eff, mu_eff) + kappa[:,n].dot(F_eff))

                # Compute the marginal probability
                lps[v] += np.linalg.slogdet(Sigma_post)[1]
                lps[v] -= np.linalg.slogdet(Sigma_eff)[1]
                lps[v] += mu_post.T.dot(np.linalg.solve(Sigma_post, mu_post))
                lps[v] -= mu_eff.T.dot(np.linalg.solve(Sigma_eff, mu_eff))
                lps[v] += v * np.log(rho) + (1-v) * np.log(1-rho)

            # Sample from the marginal probability
            ps = np.exp(lps - logsumexp(lps))
            assert np.allclose(ps.sum(), 1.0)
            A[m,n] = np.random.rand() < ps[1]

def resample_W_b():
    # Compute kappa for a bernoulli model
    kappa = S - 0.5

    for n in xrange(N):
        Aeff = np.concatenate(([1], A[:,n])).astype(np.bool)

        # Get effective params for nonzero A's
        F_eff = F_full[:, Aeff]
        mu_eff = mu_full[Aeff]
        Sigma_eff = Sigma_full[np.ix_(Aeff, Aeff)]

        # Compute the posterior parameters of W
        Lambda_post = np.linalg.inv(Sigma_eff) + (F_eff * omega[:,n][:,None]).T.dot(F_eff)
        Sigma_post = np.linalg.inv(Lambda_post)
        mu_post = Sigma_post.dot(np.linalg.solve(Sigma_eff, mu_eff) + kappa[:,n].dot(F_eff))

        Wbn = np.random.multivariate_normal(mu_post, Sigma_post)

        # Set bias and weights
        b[n] = Wbn[0]
        W[A[:,n],n] = Wbn[1:]
        W[~A[:,n],n] = mu_w + np.sqrt(sigmasq_w) * np.random.randn((~A[:,n]).sum())
        # W[~A[:,n],n] = 0

    # Update psi
    psi[:,:] = 0
    psi[:,:] += b
    psi[:,:] += F.dot(A*W)

if __name__ == "__main__":
    do_plot = False
    do_geweke = True
    N_samples = 10000

    # Define a simple model
    N = 2
    T = 50
    T_plot = 100

    # Observation model is negative binomial with
    # parameters xi and sigma(psi)
    xi = 10.

    # Basis is cosine
    basis = CosineBasis(1, 1, 2)

    # Bias prior
    mu_b = -1.
    sigmasq_b = 1.
    b = mu_b + np.sqrt(sigmasq_b) * np.random.randn(N)
    b_true = b.copy()

    # Network prior
    rho = 0.5
    mu_w = 0.
    sigmasq_w = 0.5**2

    # Combine the bias and weight priors into one
    mu_full = np.concatenate(([mu_b], mu_w * np.ones(N)))
    Sigma_full = np.diag(np.concatenate(([sigmasq_b], sigmasq_w* np.ones(N))))

    # Sample a network
    A = np.random.rand(N,N) < rho
    W = mu_w + np.sqrt(sigmasq_w) * np.random.randn(N,N)

    A_true = A.copy()
    W_true = W.copy()

    # Sample a spike train
    S, psi = sample_spiketrain(T, N, basis, A, W, b)
    # Filter the spike train
    F = basis.convolve_with_basis(S)[:,:,0]
    F_full = np.concatenate((np.ones((T,1)), F), axis=1)
    assert np.allclose(psi, b + F.dot(A*W))

    # Plot the spike train
    if do_plot:
        plt.ion()
        plt.figure()
        lns = []
        for n in xrange(N):
            plt.subplot(N,1,n+1)
            lns.append(plt.plot(psi[:T_plot,n], 'r')[0])
            plt.plot(psi[:T_plot,n], 'b')
            spks = np.where(S[:T_plot,n])[0]
            plt.plot(spks, np.ones_like(spks), 'ko', markerfacecolor="k")

            plt.ylim((min(0.9, psi.min()-0.1), max(1.1, psi.max()+0.1)))
        plt.show()


    # Do some inference
    # Instantiate the auxiliary variables
    omega = np.zeros_like(psi)

    num_threads = ppg.get_omp_num_threads()
    seeds = np.random.randint(2**16, size=num_threads)
    ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

    # Collect samples
    b_samples = []
    W_samples = []
    A_samples = []
    psi_samples = []

    for itr in xrange(N_samples):
        print "Iteration ", itr
        resample_omega()
        resample_A()
        resample_W_b()

        b_samples.append(b.copy())
        W_samples.append(W.copy())
        A_samples.append(A.copy())
        psi_samples.append(psi.copy())

        # Geweke test
        if do_geweke:
            # Sample a new spike train
            S, psi = sample_spiketrain(T, N, basis, A, W, b)
            F = basis.convolve_with_basis(S)[:,:,0]
            F_full = np.concatenate((np.ones((T,1)), F), axis=1)

        # Update plots
        if do_plot:
            for n in xrange(N):
                lns[n].set_data(np.arange(T_plot), psi[:T_plot,n])
            plt.pause(0.001)

    # Plot the sample mean
    if do_plot:
        psi_mean = np.array(psi_samples).mean(0)
        for n in xrange(N):
            plt.subplot(N,1,n+1)
            plt.plot(np.arange(T_plot), psi_mean[:T_plot,n], 'r', lw=2)

    A_mean = np.array(A_samples).mean(0)
    A_std  = np.array(A_samples).std(0)
    W_mean = np.array(W_samples).mean(0)
    W_std  = np.array(W_samples).std(0)
    b_mean = np.array(b_samples).mean(0)
    b_std  = np.array(b_samples).std(0)

    if not do_geweke:
        print "A_true: ", A_true
    print "A_mean: ", A_mean
    print "A_std:  ", A_std

    if not do_geweke:
        print "W_true: ", W_true * A_true
    print "W_mean: ", W_mean
    print "W_std:  ", W_std

    if not do_geweke:
        print "b_true: ", b_true
    print "b_mean: ", b_mean
    print "b_std:  ", b_std
