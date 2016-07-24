# distutils: extra_compile_args = -O3 -w
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

from cython cimport floating
from libc.math cimport sqrt

##########
#  BLAS  #
##########

# http://www.netlib.org/blas/

ctypedef int drotg_t(
    double *da, double *db, double *c ,double *s
    ) nogil
cdef drotg_t *drotg

ctypedef int drot_t(
    int *n, double *dx, int *incx, double *dy, int *incy, double *c, double *s
    ) nogil
cdef drot_t *drot

ctypedef int srotg_t(
    float *da, float *db, float *c ,float *s
    ) nogil
cdef srotg_t *srotg

ctypedef int srot_t(
    int *n, float *dx, int *incx, float *dy, int *incy, float *c, float *s
    ) nogil
cdef srot_t *srot

# Compute Cholesky updates and downdates
cdef inline void _chol_update(int n, floating *R, floating *z) nogil:
    """
    Cholesky update from R = chol(A) to R' = chol(A + zz^T)
    :param n: R is nxn
    :param R: R, matrix to be updated
    :param z: vector to be updated with
    :return:
    """
    cdef int k
    cdef int inc = 1
    cdef floating a, b, c, s
    if floating is double:
        for k in range(n):
            a, b = R[k*n+k], z[k]
            drotg(&a,&b,&c,&s)
            drot(&n,&R[k*n],&inc,&z[0],&inc,&c,&s)
    else:
        for k in range(n):
            a, b = R[k*n+k], z[k]
            srotg(&a,&b,&c,&s)
            srot(&n,&R[k*n],&inc,&z[0],&inc,&c,&s)

cdef inline void _chol_downdate(int n, floating *R, floating *z) nogil:
    cdef int k, j
    cdef floating rbar
    for k in range(n):
        rbar = sqrt((R[k*n+k] - z[k])*(R[k*n+k] + z[k]))
        for j in range(k+1,n):
            R[k*n+j] = (R[k*n+k]*R[k*n+j] - z[k]*z[j]) / rbar
            z[j] = (rbar*z[j] - z[k]*R[k*n+j]) / R[k*n+k]
        R[k*n+k] = rbar