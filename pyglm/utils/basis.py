import numpy as np
import scipy.linalg
import scipy.signal as sig

def convolve_with_basis(S, basis):
    """
    Convolve each column of the event count matrix with this basis
    :param S:     TxN matrix of inputs.
                  T is the number of time bins
                  N is the number of input dimensions.
    :return: TxNxB tensor of inputs convolved with bases
    """
    # TODO: Check that basis is filtered causally
    (T,N) = S.shape
    (R,B) = basis.shape

    # Concatenate basis with a layer of ones
    basis = np.vstack((np.zeros((1, B)), basis))

    # Initialize array for filtered stimulus
    F = np.empty((T,N,B))

    # Compute convolutions fo each basis vector, one at a time
    for b in np.arange(B):
        F[:,:,b] = sig.fftconvolve(S,
                                   np.reshape(basis[:,b],(R+1,1)),
                                   'full')[:T,:]

    # Check for positivity
    if np.amin(basis) >= 0 and np.amin(S) >= 0:
        np.clip(F, 0, np.inf, out=F)
        assert np.amin(F) >= 0, "convolution should be >= 0"

    return F

def interpolate_basis(basis, dt, dt_max,
                      norm=True, allow_instantaneous=False):
    # Interpolate basis at the resolution of the data
    L,B = basis.shape
    t_int = np.arange(0.0, dt_max, step=dt)
    t_bas = np.linspace(0.0, dt_max, L)

    ibasis = np.zeros((len(t_int), B))
    for b in np.arange(B):
        ibasis[:,b] = np.interp(t_int, t_bas, basis[:,b])

    # Normalize so that the interpolated basis has volume 1
    if norm:
        # ibasis /= np.trapz(ibasis,t_int,axis=0)
        ibasis /= (dt * np.sum(ibasis, axis=0))

    if not allow_instantaneous:
        # Typically, the impulse responses are applied to times
        # (t+1:t+R). That means we need to prepend a row of zeros to make
        # sure the basis remains causal
        ibasis = np.vstack((np.zeros((1,B)), ibasis))

    return ibasis


def cosine_basis(B,
                 L=100,
                 orth=False,
                 norm=True,
                 n_eye=0,
                 a=1.0/120,
                 b=0.5):
    """
    Create a basis of raised cosine tuning curves
    """
    # Number of cosine basis functions
    n_cos = B - n_eye
    assert n_cos >= 0 and n_eye >= 0


    # The first n_eye basis elements are identity vectors in the first time bins
    basis = np.zeros((L,B))
    basis[:n_eye,:n_eye] = np.eye(n_eye)

    # The remaining basis elements are raised cosine functions with peaks
    # logarithmically warped between [n_eye*dt:dt_max].
    nlin = lambda t: np.log(a*t+b)      # Nonlinearity
    u_ir = nlin(np.arange(L))           # Time in log time
    ctrs = u_ir[np.floor(np.linspace(n_eye,(L/2.0),n_cos)).astype(np.int)]
    if len(ctrs) == 1:
        w = ctrs/2
    else:
        w = (ctrs[-1]-ctrs[0])/(n_cos-1)    # Width of the cosine tuning curves

    # Basis function is a raised cosine centered at c with width w
    basis_fn = lambda u,c,w: (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(u-c)*np.pi/w/2.0)))+1)/2.0
    for i in np.arange(n_cos):
        basis[:,n_eye+i] = basis_fn(u_ir,ctrs[i],w)


    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if orth:
        basis = scipy.linalg.orth(basis)
    elif norm:
        # We can only normalize nonnegative bases
        if np.any(basis<0):
            raise Exception("We can only normalize nonnegative impulse responses!")

        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [L,1]) / (1.0/L)

    return basis