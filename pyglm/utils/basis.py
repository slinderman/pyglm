import os

import numpy as np
import scipy.linalg

class Basis:

    def __init__(self, B, dt, dt_max, prms):
        self.B = B
        self.dt = dt
        self.dt_max = dt_max
        self.params = prms
        self.basis = interpolate_basis(create_basis(prms), self.dt, self.dt_max, self.params['norm'])
        assert self.basis.shape[1] == self.B
        self.L = self.basis.shape[0]

    def convolve_with_basis(self, S):
        """
        Convolve an input matrix S (e.g. a spike train or a stimulus matrix)
        :param S: TxN matrix whose columns are to be filtered
        :param dt: the bin size for each row of S
        :param dt_max: the length of the basis
        :return: a list of length N where each entry is a TxD matrix of S[:,n]
                 convolved with the basis
        """
        T, N = S.shape

        # Filter the spike train
        filtered_S = []
        for n in range(N):
            Sn = S[:,n].reshape((-1,1))
            fS = convolve_with_basis(Sn, self.basis)

            # Flatten this manually to be safe
            # (there's surely a way to do this with numpy)
            (nT,Nc,Nb) = fS.shape
            assert Nc == 1 and Nb==self.B, \
                "ERROR: Convolution with matrix " \
                "resulted in incorrect shape: %s" % str(fS.shape)
            filtered_S.append(fS[:,0,:])

        return filtered_S

def interpolate_basis(basis, dt, dt_max, norm=True):
    # Interpolate basis at the resolution of the data
    L,B = basis.shape
    t_int = np.arange(0.0, dt_max, step=dt)
    t_bas = np.linspace(0.0, dt_max, L)

    ibasis = np.zeros((len(t_int), B))
    for b in np.arange(B):
        ibasis[:,b] = np.interp(t_int, t_bas, basis[:,b])

    # Normalize so that the interpolated basis has volume 1
    if norm:
        ibasis /=  np.trapz(ibasis,t_int,axis=0)

    return ibasis


def create_basis(prms):
    """ Create a basis for impulse response functions
    """
    typ = prms['type'].lower()
    # if type == 'exp':
    #     basis = create_exp_basis(prms)
    if typ == 'cosine':
        basis = create_cosine_basis(prms)
    # elif type == 'gaussian':
    #     basis = create_gaussian_basis(prms)
    # elif type == 'identity' or type == 'eye':
    #     basis = create_identity_basis(prms)
    # elif type == 'file':
    #     if os.path.exists(prms["fname"]):
    #         basis = load_basis_from_file(prms['fname'])
    else:
        raise Exception("Unrecognized basis type: %s", typ)
    return basis

def create_cosine_basis(prms):
    """
    Create a basis of raised cosine tuning curves
    """
    n_pts = prms['L']             # Number of points at which to evaluate the basis
    n_eye = prms['n_eye']         # Number of identity basis functions
    n_cos = prms['n_bas']         # Number of cosine basis functions'

    n_bas = n_eye + n_cos
    basis = np.zeros((n_pts,n_bas))
    
    # The first n_eye basis elements are identity vectors in the first time bins
    basis[:n_eye,:n_eye] = np.eye(n_eye)
    
    # The remaining basis elements are raised cosine functions with peaks
    # logarithmically warped between [n_eye*dt:dt_max].
    
    a = prms['a']                          # Scaling in log time
    b = prms['b']                          # Offset in log time
    nlin = lambda t: np.log(a*t+b)      # Nonlinearity
    u_ir = nlin(np.arange(n_pts))       # Time in log time
    ctrs = u_ir[np.floor(np.linspace(n_eye,(n_pts/2.0),n_cos)).astype(np.int)]
    if len(ctrs) == 1:
        w = ctrs/2
    else:
        w = (ctrs[-1]-ctrs[0])/(n_cos-1)    # Width of the cosine tuning curves
    
    # Basis function is a raised cosine centered at c with width w
    basis_fn = lambda u,c,w: (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(u-c)*np.pi/w/2.0)))+1)/2.0
    for i in np.arange(n_cos):
        basis[:,n_eye+i] = basis_fn(u_ir,ctrs[i],w)
    
    
    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if prms['orth']:
        basis = scipy.linalg.orth(basis)
    if prms['norm']:
        # We can only normalize nonnegative bases
        if np.any(basis<0):
            raise Exception("We can only normalize nonnegative impulse responses!")

        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [n_pts,1]) / (1.0/n_pts)
    
    return basis

def create_exp_basis(prms):
    """
    Create a basis of exponentially decaying functions
    """
    # Set default parameters. These can be overriden by kwargs

    # Default to a raised cosine basis
    n_pts = 100             # Number of points at which to evaluate the basis
    n_exp = prms['n_exp']   # Number of exponential basis functions
    n_eye = prms['n_eye']   # Number of identity basis functions
    n_bas = n_eye + n_exp
    basis = np.zeros((n_pts,n_bas))
    
    # The first n_eye basis elements are identity vectors in the first time bins
    basis[:n_eye,:n_eye] = np.eye(n_eye)
    
    # The remaining basis elements are exponential functions with logarithmically
    # spaced time constants
    taus = np.logspace(np.log10(1), np.log10(n_pts/3), n_exp)

    # Basis function is a raised cosine centered at c with width w
    basis_fn = lambda t,tau: np.exp(-t/tau)
    for i in np.arange(n_exp):
        basis[:,n_eye+i] = basis_fn(np.arange(n_pts),taus[i])
    
    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if prms['orth']: 
        basis = scipy.linalg.orth(basis)
    if prms['norm']:
        # We can only normalize nonnegative bases
        if np.any(basis<0):
            raise Exception("We can only normalize nonnegative impulse responses!")
        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [n_pts,1]) / (1.0/n_pts)
    
    return basis

def create_gaussian_basis(prms):
    """
    Create a basis of Gaussian bumps.
    This is primarily for spatial filters.
    """
    # Set default parameters. These can be overriden by kwargs

    # Default to a raised cosine basis
    n_gauss = prms['n_gauss']   # Tuple indicating number of Gaussian bumps along each dimension
    n_dim = len(n_gauss)
    n_eye = prms['n_eye']   # Number of identity basis functions
    n_bas = n_eye + np.prod(n_gauss)
    basis = np.zeros((n_bas,n_bas))

    # The first n_eye basis elements are identity vectors in the first time bins
    basis[:n_eye,:n_eye] = np.eye(n_eye)

    # The remaining basis functions are Gaussian bumps at intervals of 1 in each dimension
    sigma = 1
    for g1 in np.arange(np.prod(n_gauss)):
        mu = np.array(np.unravel_index(g1,n_gauss))
        for g2 in np.arange(np.prod(n_gauss)):
            x = np.array(np.unravel_index(g2,n_gauss))
            basis[n_eye+g2,n_eye+g1] = np.exp(-0.5/(sigma**2)*np.sum((x-mu)**2))


    # Basis function is a raised cosine centered at c with width w
    #basis_fn = lambda t,mu,sig: np.exp(-0.5/(sig**2)*(t-mu)**2)
    #for i in np.arange(n_gauss):
    #    basis[:,i] = basis_fn(np.arange(n_pts),mus[i],sigma)

    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if prms['orth']:
        basis = scipy.linalg.orth(basis)
    if prms['norm']:
        # We can only normalize nonnegative bases
        if np.any(basis<0):
            raise Exception("We can only normalize nonnegative impulse responses!")
        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [basis.shape[0],1])

    return basis

def create_identity_basis(prms):
    """
    Create a basis of Gaussian bumps.
    This is primarily for spatial filters.
    """
    # Set default parameters. These can be overriden by kwargs

    # Default to a raised cosine basis
    n_eye = prms['n_eye']   # Number of identity basis functions
    basis = np.eye(n_eye)

    return basis

def convolve_with_basis(stim, basis):
    """ Project stimulus onto a basis. 
    :param stim   TxD matrix of inputs. 
                  T is the number of time bins 
                  D is the number of stimulus dimensions.
    :param basis  RxB basis matrix
                  R is the length of the impulse response
                  B is the number of bases
    
    :rtype TxDxB tensor of stimuli convolved with bases
    """
    (T,D) = stim.shape
    (R,B) = basis.shape
    
    import scipy.signal as sig
    
    # First, by convention, the impulse responses are apply to times
    # (t-R:t-1). That means we need to prepend a row of zeros to make
    # sure the basis remains causal
    basis = np.vstack((np.zeros((1,B)),basis))

    # Initialize array for filtered stimulus
    fstim = np.empty((T,D,B))
    
    # Compute convolutions
    for b in np.arange(B):
        assert np.all(np.isreal(stim))
        assert np.all(np.isreal(basis[:,b]))
#         fstim[:,:,b] = sig.convolve2d(stim, 
#                                       np.reshape(basis[:,b],[R+1,1]), 
#                                       'full')[:T,:]
        fstim[:,:,b] = sig.fftconvolve(stim, 
                                       np.reshape(basis[:,b],[R+1,1]), 
                                       'full')[:T,:]
    
    return fstim

def convolve_with_low_rank_2d_basis(stim, basis_x, basis_t):
    """ Convolution with a low-rank 2D basis can be performed 
        by first convolving with the spatial basis (basis_x) 
        and then convolving with the temporal basis (basis_t)
    """
    (T,D) = stim.shape
    (Rx,Bx) = basis_x.shape
    (Rt,Bt) = basis_t.shape

    # Rx is the spatial "width" of the tuning curve. This should
    # be equal to the "width" of the stimulus.
    assert Rx==D, "ERROR: Spatial basis must be the same size as the stimulus"

    import scipy.signal as sig
    
    # First convolve with each stimulus filter
    # Since the spatial stimulus filters are the same width as the spatial
    # stimulus, we can just take the dot product to get the valid portion
    fstimx = np.dot(stim, basis_x)

    # Now convolve with the temporal basis.  
    # By convention, the impulse responses are apply to times
    # (t-R:t-1). That means we need to prepend a row of zeros to make
    # sure the basis remains causal
    basis_t = np.vstack((np.zeros((1,Bt)),basis_t))

    # Initialize array for the completely filtered stimulus
    fstim = np.empty((T,Bx,Bt))
    
    # Compute convolutions of the TxBx fstimx with each of the temporal bases
    for b in np.arange(Bt):
        fstim[:,:,b] = sig.fftconvolve(fstimx, 
                                       np.reshape(basis_t[:,b],[Rt+1,1]), 
                                       'full')[:T,:]
    
    return fstim

_fft_cache = []

def convolve_with_2d_basis(stim, basis, shape=['first', 'valid']):
    """ Project stimulus onto a basis.
    :param stim   TxD matrix of inputs.
                  T is the number of time bins
                  D is the number of stimulus dimensions.
    :param basis  TbxDb basis matrix
                  Tb is the length of the impulse response
                  Db is the number of basis dimensions.

    :rtype Tx1 vector of stimuli convolved with the 2D basis
    """
    (T,D) = stim.shape
    (Tb,Db) = basis.shape
    # assert D==Db, "Spatial dimension of basis must match spatial dimension of stimulus."

#    import scipy.signal as sig

    # First, by convention, the impulse responses are apply to times
    # (t-R:t-1). That means we need to prepend a row of zeros to make
    # sure the basis remains causal
    basis = np.vstack((np.zeros((1,Db)),basis))

    # Flip the spatial dimension for convolution
    # We are convolving the stimulus with the filter, so the temporal part does
    # NOT need to be flipped
    basis = basis[:,::-1]

    # Compute convolution using FFT
    if D==Db and shape[1] == 'valid':
        raise Warning("Use low rank convolution when D==Db!")

    # Look for fft_stim in _fft_cache
    fft_stim = None
    for (cache_stim, cache_fft_stim) in _fft_cache:
        if np.allclose(stim[-128:],cache_stim[-128:]) and \
           np.allclose(stim[:128],cache_stim[:128]):
            fft_stim = cache_fft_stim
            break

    if not fft_stim is None:
        fstim,_ = fftconv.fftconvolve(stim, basis, 'full',
                                      fft_in1=fft_stim)
    else:
        fstim,fft_stim,_ = fftconv.fftconvolve(stim, basis, 'full')
        _fft_cache.append((stim,fft_stim))

    # Slice the result
    assert len(shape) == 2
    if shape[0] == 'first':
        fstim = fstim[:T,:]
    else:
        raise Exception('Only supporting \'first\' slicing for dimension 0 (time)')

    if shape[1] == 'valid':
        assert Db == D, 'Dimension of basis must match that of stimuli for valid'
    elif shape[1] == 'central':
        sz = D + Db - 1
        start = (sz - D)/2
        stop = start + D
        fstim = fstim[:,start:stop]

    return fstim

def convolve_with_3d_basis(stim, basis, shape=['first', 'central', 'central']):
    """ Project stimulus onto a basis.
    :param stim   T x Dx x Dy array of inputs.
                  T is the number of time bins
                  Dx is the stimulus x dimension.
                  Dy is the stimulus y dimension.
    :param basis  Tb x Dbx x Dby basis matrix
                  Tb is the length of the impulse response
                  Dbx is the basis x dimension
                  Dby is the basis y dimension

    :rtype Tx1 vector of stimuli convolved with the 2D basis
    """
    assert stim.ndim == basis.ndim == 3
    (T,Dx,Dy) = stim.shape
    (Tb,Dbx,Dby) = basis.shape

    # First, by convention, the impulse responses are apply to times
    # (t-R:t-1). That means we need to prepend a row of zeros to make
    # sure the basis remains causal
    basis = np.concatenate((np.zeros((1,Dbx,Dby)),basis), axis=0)

    # Flip the spatial dimension for convolution
    # We are convolving the stimulus with the filter, so the temporal part does
    # NOT need to be flipped
    basis = basis[:,::-1, ::-1]

    # Compute convolution using FFT
    if Dx==Dbx and Dy==Dby and shape[1] == 'valid':
        raise Warning("Use low rank convolution when D==Db!")

    # Look for fft_stim in _fft_cache
    fft_stim = None
    for (cache_stim, cache_fft_stim) in _fft_cache:
        if np.allclose(stim[-128:],cache_stim[-128:]) and \
           np.allclose(stim[:128],cache_stim[:128]):
            fft_stim = cache_fft_stim
            break

    if not fft_stim is None:
        fstim,_ = fftconv.fftconvolve(stim, basis, 'full',
                                      fft_in1=fft_stim)
    else:
        fstim,fft_stim,_ = fftconv.fftconvolve(stim, basis, 'full')
        _fft_cache.append((stim,fft_stim))

    # Slice the result
    assert len(shape) == 3
    if shape[0] == 'first':
        fstim = fstim[:T,:,:]
    else:
        raise Exception('Only supporting \'first\' slicing for dimension 0 (time)')

    if shape[1] == 'full':
        pass
    elif shape[1] == 'central':
        sz = Dx + Dbx - 1
        start = (sz - Dx)/2
        stop = start + Dx
        fstim = fstim[:,start:stop, :]
    else:
        raise NotImplementedError('Only supporting full and central slicing for spatial dims')

    if shape[2] == 'full':
        pass
    elif shape[2] == 'central':
        sz = Dy + Dby - 1
        start = (sz - Dy)/2
        stop = start + Dy
        fstim = fstim[:,:,start:stop]
    else:
        raise NotImplementedError('Only supporting full and central slicing for spatial dims')

    return fstim


def project_onto_basis(f, basis, lam=0):
    """
    Project the function f onto the basis.
    :param f     Rx1 function
    :param basis RxB basis
    :param lam   Optional ridge regresion penalty
    :rtype Bx1 vector of basis coefficients
    """
    (R,B) = basis.shape

    assert f.shape[0]==R, "Function is not the same length as the basis!"

    # Make sure at least 2D
    if f.ndim==1:
        f = np.reshape(f,(R,1))

    # Regularize the projection
    Q = lam*np.eye(B)

    beta = np.dot(np.dot(scipy.linalg.inv(np.dot(basis.T,basis)+Q), basis.T),f)
    return beta

def test_convolve_3d():
    T = 100
    D1 = 5
    D2 = 5
    stim = np.random.randn(T,D1,D2)
    f_x = np.zeros((1,3,3))
    f_x[0,1,1] = 1.0
    f_t = np.array([[1]])
    f = np.tensordot(f_t, f_x, [1,0])


    import pdb; pdb.set_trace()
    fstim = convolve_with_3d_basis(stim, f, ['first', 'central', 'central'])
    assert np.allclose(stim[:-1,0,0], fstim[1:,0,0])

if __name__ == '__main__':
    test_convolve_3d()