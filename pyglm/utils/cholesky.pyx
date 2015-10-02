# distutils: extra_compile_args = -O3 -w
# distutils: include_dirs = pylds/utils/
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

from cython cimport floating
import numpy as np
cimport numpy as np
from cholesky cimport _chol_update, _chol_downdate

import scipy.linalg.blas
import scipy.linalg.lapack

from scipy.linalg import solve_triangular

cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL

cdef:
    # BLAS
    srotg_t *srotg = <srotg_t*>f2py_pointer(scipy.linalg.blas.srotg._cpointer)
    srot_t *srot = <srot_t*>f2py_pointer(scipy.linalg.blas.srot._cpointer)
    drotg_t *drotg = <drotg_t*>f2py_pointer(scipy.linalg.blas.drotg._cpointer)
    drot_t *drot = <drot_t*>f2py_pointer(scipy.linalg.blas.drot._cpointer)



cpdef _python_chol_update(floating[:,::1] R, floating[::1] z):
    """
    Update an upper triangular Cholesky decomposition, R = chol(A)
    to be R = chol(A + zz^T)

    :param R:
    :param z:
    :return:
    """
    N = R.shape[0]

    # Do the cholesky update on the upper triangular matrix Rhat
    _chol_update(N, &R[0,0], &z[0])

    return np.asarray(R)

cpdef _python_chol_downdate(floating[:,::1] R, floating[::1] z):
    """
    Downdate an upper triangular Cholesky decomposition, R = chol(A)
    to be R = chol(A - zz^T)

    :param R:
    :param z:
    :return:
    """
    N = R.shape[0]

    # Do the cholesky update on the upper triangular matrix Rhat
    _chol_downdate(N, &R[0,0], &z[0])

    return np.asarray(R)


def chol_update(floating[:,::1] L, floating[::1] z, lower=True):
    """
    Update triangular Cholesky for L = chol(A) to L' = chol(A + zz^T)
    By default assume L is lower triangular
    """
    if lower:
        return _python_chol_update(L.T.copy(), z).T
    else:
        return _python_chol_update(L, z)


def chol_downdate(floating[:,::1] L, floating[::1] z, lower=True):
    """
    Update triangular Cholesky for L = chol(A) to L' = chol(A + zz^T)
    By default assume L is lower triangular
    """
    if lower:
        return _python_chol_downdate(L.T.copy(), z).T
    else:
        return _python_chol_downdate(L, z)

def chol_add_row_to_bottom(Lprev, B, C, out=None):
    """
    Compute L* = chol(A*)
    where
    A* = [[A,   B],
          [B^T, C]]

    We know
    L* = [[L, 0],
          [E, F]]

    and that L = chol(A). By math,
    E = L^{-1} B
    F = C - E^T E

    :param Lprev:  NxN Cholesky decomposition of A
    :param B:      NxD matrix to append to A
    :param C:      DxD matrix to append to A
    :param out:    N+D x N+D output matrix or None
    """
    if B.ndim == 1:
        B = B[:,None]

    if np.isscalar(C):
        C = np.array([[C]])
    elif C.ndim == 1:
        C = C[:,None]

    N, D = Lprev.shape[0], B.shape[1]
    assert Lprev.shape[1] == N
    assert B.shape == (N,D)
    assert C.shape == (D,D)

    if out is None:
        out = np.zeros((N+D, N+D))

    E = solve_triangular(Lprev, B, lower=True).T

    # TODO: Use cholesky downdate
    F = np.linalg.cholesky(C - E.dot(E.T))

    out[:N,:N] = Lprev
    out[N:,:N] = E
    out[N:,N:] = F
    return out

def chol_add_row(Lprev, start, BDE, out=None):
    """
    Compute L* = chol(A*) given L = chol(A)
    where
    A* = [[A,    B,    C],
          [B.T,  D,    E],
          [C.T,  E.T,  F]]

    A = [[A,    C],
         [C.T,  F]]

    and
    L = [[L, 0],   = chol(A),
         [P, R]]

    Let
    L* = [[L*, 0, 0],
          [J*, K*, 0],
          [P*, Q*, R*]]

    We can see that L = chol(A). Also,
    L*P* = C = LP                       -> P* = P
    L*J* = B                            -> J* = L^{-1} B
    J*J* + K*K* = D                     -> K* = chol(D-J*J*)
    P*J* + K*Q* = E                     -> Q* = K*^{-1} (E-P*J*)
    P*P* + Q*Q* + R*R* = F = PP + RR    -> R* = chol(RR - Q*Q*)

    :param Lprev:  NxN Cholesky decomposition of A
    :param start:  Start index to add new entries to A
    :param BDE:    (Nxp) array of rows to insert into A
    :param out:    N+p x N+p output matrix or None
    """
    N = Lprev.shape[0]
    p = BDE.shape[0]
    assert BDE.shape == (N,p)

    # Get B, D, and E
    B = BDE[:start,:]
    D = BDE[start:start+p,:]
    E = BDE[start+p:,:]

    # Get L, P, and R
    L = Lprev[:start, :start]
    P = Lprev[start:, :start]
    R = Lprev[start:, start:]

    # Compute the missing pieces
    # TODO: Double check the transposes
    J = solve_triangular(L, B)
    K = np.linalg.cholesky(D-J.dot(J.T))
    Q = solve_triangular(K, E.T - P.dot(J))
    R = np.linalg.cholesky(R.dot(R.T) - Q.dot(Q.T))

    # Put them into the output
    if out is None:
        out = np.zeros((N+p, N+p))

    out[:start, :start] = L
    out[start:start+p,:start] = J
    out[start:start+p,start:start+p] = K
    out[start+p:,:start] = P
    out[start+p:,start:start+p] = Q
    out[start+p:,start+p:] = R

    return out

def chol_remove_row(Lprev, start, stop, out=None):
    """
    Compute L = chol(A) given L* = chol(A*)
    where
    A* = [[A,    B,    C],
          [B.T,  D,    E],
          [C.T,  E.T,  F]]

    We know
    L* = [[L*, 0, 0],
          [J*, K*, 0],
          [P*, Q*, R*]]

    and we want
    L = [[L, 0],   = chol([[A, C],
         [P, R]]           [C, F]]

    We can see that L = chol(A). By math,
    LP = C = L*P*  -> P = P*
    and
    P*P* + Q*Q* + R*R* = F = PP + RR
    which implies
    R = chol(Q*Q* + R*R*)

    :param Lprev:  N+DxN+D Cholesky decomposition of A
    :param slc:    slice of size D to remove from A
    :param out:    NxN output matrix or None
    """
    NpD = Lprev.shape[0]
    D = stop - start
    N = NpD - D

    if out is None:
        out = np.zeros((N, N))

    L = Lprev[:start, :start]
    P = Lprev[stop:, :start]
    Qs = Lprev[stop:, start:stop]
    Rs = Lprev[stop:, stop:]

    if stop < NpD:
        # TODO: Use cholesky update
        R = np.linalg.cholesky(Qs.dot(Qs.T) + Rs.dot(Rs.T))
    else:
        R = np.array([])

    # Fill in the output matrix
    out[:start, :start] = L
    out[start:, :start] = P
    out[start:, start:] = R

    return out
