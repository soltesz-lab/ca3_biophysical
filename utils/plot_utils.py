import numpy as np
import matplotlib
import matplotlib.pyplot as plt

    
def plot_spikes(spike_times, title, start, finish, gids=None):
    
    if gids is None:
        gids = sorted(spike_times.keys())
    frs = []
    plt.figure(figsize=(12,2))
    i = 0
    
    for gid in gids:
        sts = spike_times[gid]
        sts = np.asarray(sts)
        plt.vlines(np.asarray(sts), i+0.5, i+1.5)
        sts_chop = sts[np.where( (sts>=start) & (sts<=finish)) [0]]
        frs.append(float(len(sts_chop)) / (finish-start) * 1000.)
        i += 1
    plt.xlim([start, finish])
    plt.title('%s fr. mean: %0.3f. std: %0.3f' % (title, np.mean(frs), np.std(frs)))
    plt.show()
    return


def plot_spikes_with_density(spike_times, title, start, finish, gids=None, color='k', binsize=150):
    
    if gids is not None:
        temp_spike_times = []
        for gid in gids:
            if gid in spike_times:
                temp_spike_times.append(spike_times[gid])
        spike_times = temp_spike_times
    frs = []
    fig, ax = plt.subplots(figsize=(12,2))
    for (i,sts) in enumerate(spike_times):
        sts = np.asarray(sts)
        ax.vlines(np.asarray(sts), i+0.5, i+1.5, color=color)
        sts_chop = sts[np.where( (sts>=start) & (sts<=finish)) [0]]
        frs.append(float(len(sts_chop)) / (finish-start) * 1000.)
    ax.set_xlim([start, finish])
    #ax.set_title('%s fr: %0.3f' % (title, np.mean(frs)))
    
    ax2 = ax.twinx()
    
    spike_bins = np.arange(start, finish, binsize)
    sts_flat = np.concatenate(spike_times)
    sts_chop = sts_flat[np.where( (sts_flat>=start) & (sts_flat<=finish)) [0]]
    nspikes, spike_edges = np.histogram(sts_chop, bins=spike_bins)
    ax2.plot(spike_edges[:-1], nspikes, color='r', alpha=0.8)
    #sns.kdeplot(sts_chop, ax=ax2, bw_method=0.01)
    ax2.set_xlim([start, finish])
    return nspikes
