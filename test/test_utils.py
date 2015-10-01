import numpy as np

from pyglm.utils.utils import chol_remove_row, chol_add_row

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
