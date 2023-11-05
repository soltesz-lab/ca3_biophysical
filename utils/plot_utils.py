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
