"""
Analyze the inferred locations under the eigenmodel
"""
import numpy as np
import os
import gzip
import cPickle
import matplotlib.pyplot as plt

from hips.plotting.colormaps import harvard_colors

def analyze_eigenmaps():
    ###########################################################
    # Load the results of fitting with the Eigenmodel
    ###########################################################
    base_path = os.path.join("data", "rgc", "rgc_60T")
    results_path = base_path + ".eigen_fit.gibbs.pkl.gz"
    with gzip.open(results_path, 'r') as f:
        samples, lps, plls, timestamps = cPickle.load(f)

    ###########################################################
    # Get the latent location samples
    ###########################################################
    N_samples = len(samples)

    # Rotate the samples since they are invariant to rotation
    print "Aligning samples"
    F_star = samples[-1].network.F
    F_star = approximate_rgc_locs()
    lmbda_star = samples[-1].network.lmbda
    F_samples = []
    for s,smpl in enumerate(samples):
        R = smpl.network.adjacency_dist.\
            compute_optimal_rotation(F_star, lmbda_star)
        F_samples.append(smpl.network.F.dot(R))

        # print "rho: %.3f" % (smpl.weight_model.A.sum() / 27.0**2)

    F_samples = np.array(F_samples)

    ###########################################################
    # Scatter plot some of the locations
    ###########################################################
    print "Plotting..."
    offset = N_samples // 2
    subsmpl = 5

    plt.figure()
    plt.ion()
    plt.subplot(121)
    plt.show()
    colors = harvard_colors()
    for n in xrange(16):
        color = colors[ n % len(colors)]
        plt.plot(F_samples[offset::subsmpl,n,0],
                 F_samples[offset::subsmpl,n,1],
                 linestyle="none",
                 color=color,
                 marker='$%d$' % (n+12),
                 markerfacecolor=color,
                 alpha=0.5)

        plt.title("OFF Cell Embedding")

        plt.pause(0.001)
        raw_input("Press enter to continue")

    # Plot the ON cells
    plt.subplot(122)
    for n in xrange(16, 27):
        color = colors[ n % len(colors)]
        plt.plot(F_samples[offset::subsmpl,n,0],
                 F_samples[offset::subsmpl,n,1],
                 linestyle="none",
                 color=color,
                 marker='$%d$' % (n-16+1),
                 markerfacecolor=color,
                 alpha=0.5)

        plt.title("ON Cell Embedding")

        plt.pause(0.001)
        raw_input("Press enter to continue")


    plt.ioff()


def approximate_rgc_locs():
    """
    Approximate the RGC locations given Figure 2 of Pillow 2008
    :return:
    """
    on_locs = np.zeros((11,2))
    on_locs[0,:]  = [0.5, 2.0]
    on_locs[1,:]  = [0.9, 0.9]
    on_locs[2,:]  = [1.5, 0.2]
    on_locs[3,:]  = [1.5, 1.5]
    on_locs[4,:]  = [1.5, 2.5]
    on_locs[5,:]  = [1.0, 3.5]
    on_locs[6,:]  = [2.0, 3.5]
    on_locs[7,:]  = [2.5, 2.0]
    on_locs[8,:]  = [2.5, 1.0]
    on_locs[9,:]  = [3.5, 1.5]
    on_locs[10,:] = [3.3, 3.0]

    # Center the ON locs
    on_locs -= np.array([[2,2]])

    # Manually approximate the OFF locations
    off_locs = np.zeros((16,2))
    off_locs[0,:]  = [0.5, 2.5]
    off_locs[1,:]  = [0.5, 3.5]
    off_locs[2,:]  = [0.6, 1.5]
    off_locs[3,:]  = [1.5, 2.2]
    off_locs[4,:]  = [1.5, 3.0]
    off_locs[5,:]  = [1.5, 3.5]
    off_locs[6,:]  = [2.5, 3.5]
    off_locs[7,:]  = [2.5, 3.0]
    off_locs[8,:]  = [1.7, 1.3]
    off_locs[9,:]  = [1.0, 0.8]
    off_locs[10,:] = [2.5, 2.5]
    off_locs[11,:] = [1.7, 0.5]
    off_locs[12,:] = [2.5, 1.5]
    off_locs[13,:] = [2.8, 0.5]
    off_locs[14,:] = [3.5, 2.5]
    off_locs[15,:] = [3.5, 1.3]

    # Center the OFF locs
    off_locs -= np.array([[2,2]])

    # Stack up the locations
    L = np.vstack((off_locs, on_locs))
    return L

analyze_eigenmaps()
