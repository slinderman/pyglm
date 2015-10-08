"""
This is a dummy module to facilitate joblib Parallel
 resampling of the adjacency matrix. It just keeps
 pointers to the global variables so that the parallel
 processes can reference them. No copying needed!
"""
import numpy as np
from scipy.linalg.lapack import dpotrs
from pyglm.utils.utils import sample_gaussian

# Set these as module level global variables
augmented_data = None
weight_model = None
bias_model = None
network = None
P = None
N = None
B = None


def _parallel_collapsed_resample_column(n):
    # Compute the prior and posterior sufficient statistics of W
    J_prior, h_prior = _prior_sufficient_statistics(n)
    J_lkhd, h_lkhd = _lkhd_sufficient_statistics(n, augmented_data)
    J_post = J_prior + J_lkhd
    h_post = h_prior + h_lkhd

    An = _collapsed_resample_A(n, J_prior, h_prior, J_post, h_post)
    bn, Wn = _collapsed_resample_W_b(An, J_post, h_post)
    return bn, An, Wn

def _collapsed_resample_A(n, J_prior, h_prior, J_post, h_post):
    """
    Resample the presence or absence of a connection (synapse)
    """
    A_col = weight_model.A[:,n].copy()

    # First sample A[:,n] -- the presence or absence of a connections from
    # presynaptic neurons
    perm = np.random.permutation(N)

    ml_prev = _marginal_likelihood(A_col, J_prior, h_prior, J_post, h_post)

    for m in perm:
        # Compute the marginal prob with and without A[m,n]
        lps = np.zeros(2)

        # We already have the marginal likelihood for the current value of A
        # We just need to add the prior
        v_prev = A_col[m]
        lps[v_prev] += ml_prev
        lps[v_prev] += v_prev * np.log(P[m,n]) + (1-v_prev) * np.log(1-P[m,n])

        # Now compute the posterior stats for 1-v
        v_new = 1 - v_prev
        A_col[m] = v_new

        ml_new = _marginal_likelihood(A_col, J_prior, h_prior, J_post, h_post)

        lps[v_new] += ml_new
        lps[v_new] += v_new * np.log(P[m,n]) + (1-v_new) * np.log(1-P[m,n])

        # Sample from the marginal probability
        max_lps = max(lps[0], lps[1])
        se_lps = np.sum(np.exp(lps-max_lps))
        lse_lps = np.log(se_lps) + max_lps
        ps = np.exp(lps - lse_lps)

        # ps = np.exp(lps - logsumexp(lps))
        # assert np.allclose(ps.sum(), 1.0)
        v_smpl = np.random.rand() < ps[1]
        A_col[m] = v_smpl

        # Cache the posterior stats and update the matrix objects
        if v_smpl != v_prev:
            ml_prev = ml_new

    return A_col

def _collapsed_resample_W_b(A_col, J_post, h_post):
    """
    Resample the weight of a connection (synapse)
    """
    Aeff = np.concatenate(([1], np.repeat(A_col, B))).astype(np.bool)
    Jp = J_post[np.ix_(Aeff, Aeff)]
    hp = h_post[Aeff]

    # Sample in mean and covariance (standard) form
    # mup = np.linalg.solve(Jp, hp)
    # Sigp = np.linalg.inv(Jp)
    # Wbn = np.random.multivariate_normal(mup, Sigp)

    # Sample in information form
    Wbn = sample_gaussian(J=Jp, h=hp)

    # Set bias and weights
    bn = Wbn[0]
    W_col = np.zeros((N,B))
    W_col[A_col.astype(np.bool)] = Wbn[1:].reshape((-1,B))

    return bn, W_col

def _marginal_likelihood(A_col, J_prior, h_prior, J_post, h_post):
    """
    Compute the marginal likelihood as the ratio of log normalizers
    """
    Aeff = np.concatenate(([1], np.repeat(A_col, B))).astype(np.bool)

    # Extract the entries for which A=1
    J0 = J_prior[np.ix_(Aeff, Aeff)]
    h0 = h_prior[Aeff]
    Jp = J_post[np.ix_(Aeff, Aeff)]
    hp = h_post[Aeff]

    # Compute the marginal likelihood
    L0 = np.linalg.cholesky(J0)
    Lp = np.linalg.cholesky(Jp)

    ml = 0
    ml -= np.sum(np.log(np.diag(Lp)))
    ml += np.sum(np.log(np.diag(L0)))
    ml += 0.5*hp.T.dot(dpotrs(Lp, hp, lower=True)[0])
    ml -= 0.5*h0.T.dot(dpotrs(L0, h0, lower=True)[0])

    return ml


def _prior_sufficient_statistics(n):
    mu_b    = bias_model.mu_0
    sigma_b = bias_model.sigma_0

    mu_w    = network.weights.Mu[:, n, :]
    Sigma_w = network.weights.Sigma[:, n, :, :]

    # Create a vector mean and a block diagonal covariance matrix
    from scipy.linalg import block_diag
    mu_full = np.concatenate(([mu_b], mu_w.ravel()))
    Sigma_full = block_diag([[sigma_b]], *Sigma_w)

    # Make sure they are the right size
    assert mu_full.ndim == 1
    K = mu_full.shape[0]
    assert Sigma_full.shape == (K,K)

    # Compute the information form
    J_prior = np.linalg.inv(Sigma_full)
    h_prior = J_prior.dot(mu_full)

    return J_prior, h_prior

def _lkhd_sufficient_statistics(n, augmented_data):
    """
    Compute the full likelihood sufficient statistics as if all connections
     were present.
    """
    # Compute the sufficient statistics of the likelihood
    # These will be the same shape as those of the prior
    D = 1 + N * B
    J_lkhd = np.zeros((D, D))
    h_lkhd = np.zeros(D)

    # Compute the posterior sufficient statistics
    for data in augmented_data:
        T = data["T"]
        omega = data["omega"]
        kappa = data["kappa"]

        F_flat = data["F_full"]
        assert F_flat.shape == (T,N*B+1)

        # The likelihood terms will be dense
        # h_lkhd is a 1xT vector times a T x NB matrix
        # We will only need a subset of the resulting NB vector
        J_lkhd += (F_flat * omega[:,n][:,None]).T.dot(F_flat)
        h_lkhd  += kappa[:,n].dot(F_flat)

    return J_lkhd, h_lkhd

