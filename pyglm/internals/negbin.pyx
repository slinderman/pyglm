# distutils: extra_compile_args = -O3 -w
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

from cython cimport integral, floating
from cython.parallel import prange
from libc.math cimport log
import numpy as np
cimport numpy as np

from scipy.special import gammaln

cpdef nb_likelihood_xi(integral[::1] S, floating[::1] P, floating[::1] xis, floating[::1] lps):
    """
    compute the likelihood of xi in a negative binomial distribution
    """
    cdef int M = xis.shape[0]
    cdef int N = S.shape[0]
    cdef int m, n, s

    cdef floating xi, gammaln_xi

    # lp = (gammaln(S[:,None]+xi) - gammaln(xi)).sum(0)
    # Simplify this as follows:
    # ln G(z-m) = ln G(z) - \sum_{k=1}^m ln(z-k)
    # => ln G(y) = ln G(y+m) - \sum_{k=1}^m ln(y+m-k)
    # => ln G(y) = ln G(y+m) - \sum_{j=0}^m-1 ln(y+j)
    # => ln G(s + xi) - ln G(xi) = \sum_{j=0}^{s-1} ln(xi + j)
    #
    # If s+xi is large, then this is well approximated by
    # ln G(s+xi) = (s+xi-0.5)ln(s+xi) -s -xi +0.5ln(2pi)
    cdef floating[::1] gammaln_xis = gammaln(xis)
    cdef double half_ln2pi = 0.5 * log(2*np.pi)

    with nogil:
        for m in prange(M):
            xi = xis[m]
            gammaln_xi = gammaln_xis[m]
            lps[m] = 0.0


            for n in range(N):
                # Add the term that depends on P
                lps[m] += xi * log(1-P[n])

                if S[n] == 0:
                    continue

                # If S + xi is large, use the approximation
                if S[n] + xi >= 3:
                    lps[m] += (S[n] + xi - 0.5) * log(S[n]+xi) - S[n] - xi + half_ln2pi
                    lps[m] -= gammaln_xi

                # Otherwise, use the direct sum
                else:
                    for s in range(S[n]):
                        lps[m] += log(xi+s)
