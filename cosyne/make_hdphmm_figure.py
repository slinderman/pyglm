import numpy as np
import matplotlib
import os, cPickle, gzip
import brewer2mpl

from hips.plotting.layout import *
from hips.plotting.colormaps import white_to_color_cmap, gradient_cmap
from hips.distributions.circular_distribution import CircularDistribution

def convert_xy_to_polar(pos, center, radius=np.Inf):
    # Convert true position to polar
    pos_c = (pos - center)
    pos_r = np.sqrt((pos_c**2).sum(axis=1))
    pos_r = np.clip(pos_r, 0, radius)
    pos_th = np.arctan2(pos_c[:,1], pos_c[:,0])

    return pos_r, pos_th

def convert_polar_to_xy(pos, center):
    # Convert true position to polar
    pos_x = center[0] + pos[:,0] * np.cos(pos[:,1])
    pos_y = center[1] + pos[:,0] * np.sin(pos[:,1])

    return pos_x, pos_y


def plot_latent_states(model, pos, center, radius,
                       figdir='figures/aas',
                      figsize=(1.35, 1.6),
                      num_states_to_plot=5):
    """
    Plot the observation vector associated with a latent state
    """
    # import pdb; pdb.set_trace()
    states_obj = model.states_list
    used_states = np.array(model._get_used_states(states_obj))
    N_used_states = len(used_states)
    # N_used_states = min(len(used_states), 4)
    raw_state_seq = states_obj[0].stateseq

    # Occupancy
    occupancy = np.zeros(len(used_states))

    # Get the observation vector
    obs_distns  = []

    # Convert the raw state seq to 1:len(used_states)
    state_seq = np.zeros_like(raw_state_seq)
    for new_state, old_state in enumerate(used_states):
        state_seq[raw_state_seq==old_state] = new_state
        occupancy[new_state] = np.sum(raw_state_seq==old_state)
        obs_distns.append(model.obs_distns[old_state])

    # Sort states by occupancy
    sorted_states = np.argsort(occupancy)[::-1]
    # occupancy = occupancy[sorted_states]

    # Get a permutation (second obs distn)
    Weff = obs_distns[sorted_states[1]].W_effective
    Weff_symm = (Weff + Weff.T)/2.
    from spclust import *
    recover_perm = find_blockifying_perm(Weff_symm,k=10,nclusters=10)

    # Plot a figure for each latent state
    bmap = brewer2mpl.get_map('Set1', 'qualitative', np.clip(N_used_states, 3, 9).astype(np.int))
    colors = bmap.mpl_colors
    for i,n in enumerate(sorted_states[:num_states_to_plot]):
        fig = plt.figure(figsize=figsize)
        color = colors[i]
        cmap = white_to_color_cmap(color)

        # Plot the firing rate vector
        # ax1 = create_axis_at_location(fig, 0.4, 0.1, 0.25, 1.3, box=True, ticks=True)
        # rates = np.array([poi.lmbda for poi in obs_distns[n]._distns])
        # vmax = 4
        # if not np.amax(rates) < vmax:
        #     print "WARNING: firing rate (%f) exceeds vmax (%f)" % (np.amax(rates), vmax)
        # N = rates.size
        # ax1.imshow(rates.reshape((N,1)),
        #           vmin=0,vmax=4,
        #           interpolation='nearest',
        #           aspect='auto',
        #           cmap=cmap)
        # # ax1.set_ylabel('${\\mathbf{\\lambda}}^{(%d)}$'%(i+1),
        # #                fontdict={'fontsize' : 9})
        # fig.text(0.01, 0.45,
        #          '${\\mathbf{\\lambda}}^{(%d)}$'%(i+1),
        #          rotation='vertical',
        #          fontdict={'fontsize' : 9})
        # ax1.set_yticks([0, 9, 19, 29, 39, 46])
        # ax1.set_yticklabels(map(str,["$n=$1", 10, 20, 30, 40, 47]),
        #                     fontsize=9)
        # ax1.set_xticks([])


        # Plot the locations of this state
        ax2 = create_axis_at_location(fig, .05, 0.1, 1.25, 1.25)
        remove_plot_labels(ax2)
        # Plot the empirical location distribution
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos[state_seq==n,0], pos[state_seq==n,1])
        cd.plot(ax=ax2, cmap=cmap, plot_data=True, plot_colorbar=False)

        # Plot the mean
        # xm,ym = cd.mean
        # ax2.plot(xm,ym,'o',
        #          markersize=8,
        #          markerfacecolor=color,
        #          markeredgecolor='k',
        #          markeredgewidth=2)

        # histogram_positions(pos[state_seq==n,:], center, radius, markersize=1, lw=1, ax=ax2, cmap=cmap)
        ax2.set_title('State %d (%.1f%%)' % (i+1, 100.*occupancy[n]/pos.shape[0]),
                      fontdict={'fontsize' : 9})

        fig.savefig(os.path.join(figdir, 'state%d.pdf' % (i+1)))
        plt.close(fig)

        # Plot the network
        cmap3 = gradient_cmap([color, [1,1,1], color])
        figsize2 = (figsize[0], figsize[1] + 0.25)
        fig2 = plt.figure(figsize=figsize2)
        ax = create_axis_at_location(fig2, .05, 0.1, 1.25, 1.25)
        obs_distns[n].plot_weighted_network(ax=ax, cmap=cmap3, perm=recover_perm)
        N = obs_distns[n].N
        # ax.set_xlabel('Post')
        ax.set_xlim(0,N)
        ax.set_xticks([])
        # ax.set_ylabel('Pre')
        ax.set_ylim(N,0)
        ax.set_yticks([])
        ax.set_title('Interaction weights')
        # plt.tight_layout()
        fig2.savefig(os.path.join(figdir, 'network%d.pdf' % (i+1)))


    # # Plot all the maps on top of one another
    fig = plt.figure(figsize=figsize)
    ax = create_axis_at_location(fig, .05, 0.1, 1.25, 1.25)
    remove_plot_labels(ax)

    import matplotlib.patches
    circle = matplotlib.patches.Circle(xy=[0,0],
                                       radius= radius,
                                       linewidth=1,
                                       edgecolor="k",
                                       facecolor="none")
    ax.add_patch(circle)

    for i,n in enumerate(sorted_states[:N_used_states]):
        relocc = occupancy[n] / np.float(np.amax(occupancy))
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos[state_seq==n,0], pos[state_seq==n,1])
        rm, thm = cd.mean
        xm,ym = convert_polar_to_xy(np.array([[rm, thm]]), [0,0])
        ax.plot(xm,ym,'o',
                 markersize=relocc*6,
                 markerfacecolor='k',
                 markeredgecolor='k',
                 markeredgewidth=1)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    ax.set_title('All states')
    fig.savefig(os.path.join(figdir,'all_states.pdf'))
    plt.close(fig)



# Load the data
dat_file = os.path.join('data', 'wilson.pkl')
with open(dat_file) as f:
    raw_data = cPickle.load(f)

pos = raw_data['pos']
center = raw_data['cp'].ravel()
radius = np.float(raw_data['r'])

# TODO: Load the results
# res_dir = os.path.join('results', 'wilson', 'runs', '1')
# res_dir = os.path.join('runs','1')
import sys
from glob import glob

res_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
filename = sorted(glob(os.path.join(res_dir,'iter*')))[-1]

with gzip.open(filename) as infile:
    hmm = cPickle.load(infile)

plot_latent_states(hmm, pos, center, radius, figdir=res_dir)

