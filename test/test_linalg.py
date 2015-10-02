import numpy as np
import scipy.linalg

from pyglm.utils.linalg import \
    chol_update, chol_downdate, chol_remove_row, chol_add_row, \
    block_inverse_add_rows, block_inverse_remove_rows

# def test_chol_update():
#     N = 5
#     X = np.random.randn(10,N)
#     A = X.T.dot(X)
#     U = scipy.linalg.cholesky(A).copy('C')
#     z = np.random.randn(N)
#
#     Uhat = np.copy(U)
#     Utrue = scipy.linalg.cholesky(A + np.outer(z,z))
#
#     chol_update(Uhat, z, lower=False)
#     assert np.allclose(Uhat.T.dot(Uhat), Utrue.T.dot(Utrue))

def test_chol_update():
    A = np.random.randn(3,3)
    A = A.dot(A.T)
    A += 4*np.eye(3)

    v = np.random.randn(3)

    L = np.linalg.cholesky(A)
    Ltilde = np.linalg.cholesky(A + np.outer(v,v))

    Ltilde2 = chol_update(L, v)
    assert np.allclose(Ltilde,Ltilde2, atol=1e-8)

def test_chol_downdate():

    A = np.random.randn(3,3)
    A = A.dot(A.T)
    A += 4*np.eye(3)

    v = np.random.randn(3)

    L = np.linalg.cholesky(A)
    Ltilde = np.linalg.cholesky(A - np.outer(v,v))

    Ltilde2 = chol_downdate(L, v)
    assert np.allclose(Ltilde,Ltilde2, atol=1e-8)

def test_chol_add_remove():
    N = 5
    X = np.random.randn(10,N)
    A = X.T.dot(X)
    L = np.linalg.cholesky(A)

    Am = A[:-1,:-1]
    bm = A[:-1,-1]
    cm = A[-1,-1]
    Lm = np.linalg.cholesky(Am)

    # Get chol by adding row
    assert np.allclose(L, chol_add_row(Lm, bm, cm))

    # Now get chol by removing a row
    def to_range(start, stop):
        return np.setdiff1d(np.arange(N), np.arange(start,stop))
    assert np.allclose(
        np.linalg.cholesky(A[np.ix_(to_range(4,5),
                                    to_range(4,5))]),
                           chol_remove_row(L,4,5))

    assert np.allclose(
        np.linalg.cholesky(A[np.ix_(to_range(1,3),
                                    to_range(1,3))]),
                           chol_remove_row(L,1,3))

def test_block_inverse_add_row():
    N = 4
    X = np.random.randn(100,N)
    A = X.T.dot(X)
    # A = np.random.randn(4,4)
    # A = A.dot(A.T)
    # A += 4*np.eye(4)
    Ainv = np.linalg.inv(A)

    # Get a subset of A and add rows
    end = 3
    Bm = A[:end,end:]
    Cm = Bm.T
    Dm = A[end:,end:]

    Pt,Qt,Rt,St = block_inverse_add_rows(Ainv[:end,:end], Bm, Cm, Dm, symm=True)
    assert np.allclose(Ainv[:end,:end], Pt, atol=1e-3)
    assert np.allclose(Ainv[:end,end:], Qt, atol=1e-3)
    assert np.allclose(Ainv[end:,:end], Rt, atol=1e-3)
    assert np.allclose(Ainv[end:,end:], St, atol=1e-3)

def test_block_inverse_remove_row():
    N = 4
    X = np.random.randn(100,N)
    A = X.T.dot(X)
    # A = np.random.randn(4,4)
    # A = A.dot(A.T)
    # A += 4*np.eye(4)
    Ainv = np.linalg.inv(A)

    # Get a subset of A and add rows
    for end in xrange(1,4):
        Am = A[:end,:end]
        Pmt = np.linalg.inv(Am)

        Pmt_tilde = block_inverse_remove_rows(Ainv, end, symm=True)
        assert np.allclose(Pmt, Pmt_tilde, atol=1e-3)


if __name__ == "__main__":
    # test_chol_add_remove()
    # test_chol_update()
    # test_chol_downdate()
    test_block_inverse_add_row()
    test_block_inverse_remove_row()