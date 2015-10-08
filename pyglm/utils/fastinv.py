"""
Fast matrix inverses when adding or removing rows.
Use the block inverse formula to do this efficiently.
"""
import numpy as np
from scipy.linalg.lapack import dpotrs
from scipy.linalg import solve, solve_triangular


from pyglm.utils.profiling import line_profiled
PROFILING = True

#TODO: Currently the indexing and slicing operations are killing performance!

class BigInvertibleMatrix(object):
    """
    Class for a big, invertible matrix. We want to
    compute the inverse of submatrices. For example,
    let
            [[A B C],
        M =  [D E F],
             [G H J]]
    We want to compute the inverse of M given the
    inverse S^{-1} of

        S = [[A B],
             [D E]].

    This can be done efficiently with the block inverse formula.

    """
    def __init__(self, M):
        """
        :param M: The big matrix whose submatrices we wish to invert.
        """
        self.M = M
        self.D = M.shape[0]
        assert M.shape == (self.D, self.D)
        # self.is_symm = np.allclose(M - M.T, 0)
        self.is_symm = False

        # Initialize our partial inverse to an empty array
        self._partial_inverse = np.zeros((0,0))
        self._partial_logdet = 0.
        self._partial_inds = np.array([], dtype=np.int)

    def update(self, partial_inds, partial_inverse, partial_logdet):
        self._partial_inds = partial_inds
        self._partial_inverse = partial_inverse
        self._partial_logdet = partial_logdet

    def refresh(self):
        """
        Recompute the inverse since numerical errors build up
        """
        Msub = self.M[np.ix_(self._partial_inds, self._partial_inds)]
        self._partial_inverse = np.linalg.inv(Msub)
        self._partial_logdet = np.linalg.slogdet(Msub)[1]

    @line_profiled
    def compute_submatrix_inverse(self, inds=None, added_inds=None, removed_inds=None):
        """
        Find the set difference between the desired indices
        and the indices of the current partial inverse.
        """
        pinds = self._partial_inds
        pinv = self._partial_inverse
        plogdet = self._partial_logdet

        # First compute the block inverse using the new inds
        added_inds = np.setdiff1d(inds, pinds, assume_unique=True)
        if added_inds.size > 0:
            Q = self.M[np.ix_(pinds, added_inds)]
            R = Q.T if self.is_symm else self.M[np.ix_(added_inds, pinds)]
            S = self.M[np.ix_(added_inds, added_inds)]
            Ainv = block_inverse_add_rows(
                pinv, Q, R, S, symm=self.is_symm)

            # Compute the partial determinant
            plogdet = plogdet + np.linalg.slogdet(S - R.dot(pinv.dot(Q)))[1]

            # Permute so that partial inds are ordered
            all_inds = np.concatenate((pinds, added_inds))
            perm = np.argsort(all_inds)

            pinv = Ainv[np.ix_(perm, perm)]
            pinds = all_inds[perm]

        # Now remove the indices that are not in pinds
        # and compute the new inverse. pinds must be a super set
        # of inds since now pinds' = pinds + (inds - pinds) >= inds
        local_kept = np.in1d(pinds, inds, assume_unique=True)
        global_kept = pinds[local_kept]
        local_removed = ~local_kept
        global_removed = pinds[local_removed]
        if np.sum(local_removed) > 0:
            # Partial determinant requires det A and det St
            # plogdet1 = plogdet + \
            #           np.linalg.slogdet(pinv[np.ix_(local_removed, local_removed)])[1]

            # Get the inverse after removing these rows
            pinv = block_inverse_remove_rows(pinv, np.where(local_removed)[0], symm=self.is_symm)

            # Compute the partial determinant
            Q = self.M[np.ix_(global_kept, global_removed)]
            R = Q.T if self.is_symm else self.M[np.ix_(global_removed, global_kept)]
            S = self.M[np.ix_(global_removed, global_removed)]
            plogdet2 = plogdet - np.linalg.slogdet(S-R.dot(pinv.dot(Q)))[1]

            # Partial determinant requires det A and det St
            # assert np.allclose(plogdet1, plogdet2, atol=1e-0)

            pinds = pinds[local_kept]
            plogdet = plogdet2

        return pinds, pinv, plogdet

### Matrix block inversion
@line_profiled
def block_inverse_add_rows(Pinv, Q, R, S, symm=False):
    """
    Compute the inverse of the matrix

            A = [[P, Q],
                 [R, S]]

    Given that we already know P^{-1}.
    We follow the notation of Numerical Recipes S2.7

    :param symm: If True, Q=R.T

    :return: A^{-1}
    """
    # Let A^{-1} = [[Pt, Qt],
    #               [Rt, St]]
    # where t is short for tilde

    # Precompute reusable pieces
    PiQ = Pinv.dot(Q)
    RPi = PiQ.T if symm else R.dot(Pinv)

    # Compute the outputs
    if symm:
        F = S-R.dot(PiQ)
        L = np.linalg.cholesky(F)
        St = dpotrs(
            L, np.eye(F.shape[0]), lower=True)[0]
        Rt = -solve_triangular(L, RPi)
        Pt = Pinv - PiQ.dot(solve_triangular(L, RPi))

        Qt = Rt.T
    else:
        St = np.linalg.inv(S - R.dot(PiQ))
        Pt = Pinv + PiQ.dot(St).dot(RPi)
        Qt = -PiQ.dot(St)
        Rt = Qt.T if symm else -St.dot(RPi)

    Ainv = np.vstack([np.hstack((Pt, Qt)),
                      np.hstack((Rt, St))])

    return Ainv

@line_profiled
def block_determinant_add_rows(Pinv, Q, R, S, symm=False):
    """
    Compute the determinant of the matrix

            A = [[P, Q],
                 [R, S]]

    Given that we already know P^{-1} and det{P}.
    We follow the notation of Numerical Recipes S2.7

    :param symm: If True, Q=R.T

    :return: A^{-1}
    """
    # Let A^{-1} = [[Pt, Qt],
    #               [Rt, St]]
    # where t is short for tilde

    # Precompute reusable pieces
    PiQ = Pinv.dot(Q)
    RPi = PiQ.T if symm else R.dot(Pinv)

    # Compute the outputs
    if symm:
        raise Exception("Broken!")
        F = S-R.dot(PiQ)
        L = np.linalg.cholesky(F)
        St = dpotrs(
            L, np.eye(F.shape[0]), lower=True)[0]
        Rt = -solve_triangular(L, RPi, lower=False)
        Pt = Pinv - PiQ.dot(solve_triangular(L, RPi))

        Qt = Rt.T
    else:
        St = np.linalg.inv(S - R.dot(PiQ))
        Pt = Pinv + PiQ.dot(St).dot(RPi)
        Qt = -PiQ.dot(St)
        Rt = Qt.T if symm else -St.dot(RPi)

    Ainv = np.vstack([np.hstack((Pt, Qt)),
                      np.hstack((Rt, St))])

    return Ainv

@line_profiled
def block_inverse_remove_rows(Ainv, removed_inds, symm=False):
    """
    Compute the inverse of the matrix P given

            A^{-1} = [[Pt, Qt],
                      [Rt, St]]

    We follow the notation of Numerical Recipes S2.7

    :param end: The index where Qt and Rt begin
    :param symm: If True, Qt=Rt.T

    :return: P^{-1}
    """
    # Permute such that the rows to be removed are at the end
    D = Ainv.shape[0]
    keep = np.setdiff1d(np.arange(D), removed_inds)
    N_keep = keep.size
    perm = np.concatenate([keep, removed_inds])
    Ainvp = Ainv[np.ix_(perm, perm)]

    # Extract the relevant pieces
    Pt = Ainvp[:N_keep, :N_keep]
    Qt = Ainvp[:N_keep, N_keep:]
    Rt = Qt.T if symm else Ainvp[N_keep:, :N_keep]
    St = Ainvp[N_keep:, N_keep:]

    # From the block inversion formula,
    # Pinv = Pt + PiQ.dot(St).dot(RPi)
    #      = Pt - Qt.dot(RPi)
    #      = Pt - Qt.dot(St^{-1}).dot(Rt)
    Pinv = Pt - Qt.dot(solve(St,Rt))

    return Pinv
