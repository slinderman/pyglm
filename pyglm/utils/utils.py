import numpy as np

def logistic(x):
    return 1./(1+np.exp(-x))

# Expand a mean vector
def expand_scalar(x, shp):
    if np.isscalar(x):
        x *= np.ones(shp)
    else:
        assert x.shape == shp
    return x

# Expand the covariance matrices
def expand_cov(c, shp):
    assert len(shp) >= 2
    assert shp[-2] == shp[-1]
    d = shp[-1]
    if np.isscalar(c):
        c = c * np.eye(d)
        tshp = np.array(shp)
        tshp[-2:] = 1
        c = np.tile(c, tshp)
    else:
        assert c.shape == shp

    return c
