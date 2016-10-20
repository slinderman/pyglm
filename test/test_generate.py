import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from pyglm.regression import SparseBernoulliRegression
from pyglm.models import NonlinearAutoregressiveModel
from pyglm.utils.basis import cosine_basis

def test_means():
    N = 2   # Number of neurons
    B = 3   # Number of basis functions
    L = 10  # Length of basis functions

    basis = cosine_basis(B, L=L) / L
    regressions = [SparseBernoulliRegression(N, B, mu_b=-2, S_b=0.1) for n in range(N)]
    model = NonlinearAutoregressiveModel(N, regressions, basis=basis)

    X, Y = model.generate(T=1000, keep=False)

    model.add_data(Y)
    Xtest = model.data_list[0][0]

    assert np.allclose(X, Xtest)

    means = model.means
    model.data_list[0] = (X, Y)
    means2 = model.means
    assert np.allclose(means, means2)

    plt.figure()
    for n in range(N):
        plt.subplot(N,1,n+1)
        plt.plot(means[0][:,n], lw=4)
        plt.plot(means2[0][:,n], lw=1)
        tn = np.where(Y[:,n])[0]
        plt.plot(tn, np.ones_like(tn), 'ko')
        plt.ylim(-0.05, 1.1)
    plt.show()


def test_basis():
    N = 2  # Number of neurons
    B = 3  # Number of basis functions
    L = 10  # Length of basis functions

    regressions = [SparseBernoulliRegression(N, B, mu_b=-2, S_b=0.1) for n in range(N)]
    model = NonlinearAutoregressiveModel(N, regressions, B=B)

    X, Y = model.generate(T=1000, keep=False)

    # Check that the lags are working properly
    for n in range(N):
        for b in range(B):
            assert np.allclose(Y[:-(b+1),n], X[(b+1):,n,b])


if __name__ == "__main__":
    test_basis()
    test_means()
