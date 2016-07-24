import pyglm.utils.fastinv
reload(pyglm.utils.fastinv)
from pyglm.utils.fastinv import *

def test_submatrix_class():
    N = 10
    X = np.random.randn(1000,N)
    A = X.T.dot(X)

    mat = BigInvertibleMatrix(A)

    # Get a random submatrix
    for itr in xrange(10):
        inds = np.where(np.random.rand(N) < 0.5)[0]
        # inds = np.arange(itr+1).astype(np.int)
        Asub = A[np.ix_(inds, inds)]

        pinds, pinv, pdet = mat.compute_submatrix_inverse(inds)
        Ainv_true = np.linalg.inv(Asub)
        Adet_true = np.linalg.slogdet(Asub)[1]

        assert np.allclose(pinv, Ainv_true, atol=1e-8)
        assert np.allclose(pdet, Adet_true, atol=1e-8)

        # Update
        mat.update(pinds, pinv, pdet)

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
    # test_block_inverse_add_row()
    # test_block_inverse_remove_row()
    test_submatrix_class()