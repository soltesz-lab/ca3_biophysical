import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import peri_event_time_histogram
from scipy.stats import gaussian_kde

font = {'family' : 'sans-serif',
        'sans-serif': 'Arial',
        'style': 'normal',
        'weight': 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

    
def plot_spikes(spike_times, title, start, finish, gids=None, figsize=(6,2), colors='k', add_title=False, events=None):
    start = start / 1000.
    finish = finish / 1000.
    if gids is None:
        gids = sorted(spike_times.keys())
    frs = []
    plt.figure(dpi=600, figsize=figsize)
    i = 0

    sts_dict = {}
    for gid in gids:
        sts = spike_times[gid]
        sts = np.asarray(sts) / 1000.
        sts_dict[gid] = sts
        plt.eventplot(np.asarray(sts).reshape((1,-1)), lineoffsets=i+0.5, 
                      orientation='horizontal',
                      colors=colors)
        sts_chop = sts[np.where( (sts>=start) & (sts<=finish)) [0]]
        frs.append(float(len(sts_chop)) / (finish-start))
        i += 1
        
    if events is not None:
        event_y = np.ones(len(events)) * (len(gids) + 4)
        plt.plot(events, event_y, marker='v', color='r', markersize=8, linestyle='none',)
        
    plt.xlim([start, finish])
    if add_title:
        plt.title('%s fr. mean: %0.3f. std: %0.3f' % (title, np.mean(frs), np.std(frs)))
    plt.show()
    return sts_dict
        

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



def plot_peri_event_time_histogram(spike_times, time_extent, normalized=False,
                                   bin_size=100,
                                   color=None, axlabel=None,
                                   label=None, ax=None):
    
    sts = np.concatenate(list(spike_times.values()))
    
    if ax is None:
        ax = plt.gca()
    if color is None:
        line, = ax.plot(sts.mean(), 0)
        color = line.get_color()
        line.remove()

    rate, bins = peri_event_time_histogram(spike_times, time_extent,
                                           bin_size=bin_size,
                                           normalized=normalized)
    ax.bar(bins[:-1] - (time_extent[0] + (time_extent[1] - time_extent[0]) / 2), rate,
           width=bin_size,
           edgecolor=color,
           linewidth=2,
           fill=False)

        

def plot_avg_peri_event_time_histogram(spike_times, event_time_ranges, 
                                       bin_size=100,
                                       color=None, axlabel=None,
                                       label=None, ax=None):
    
    sts = np.concatenate(list(spike_times.values()))
    
    if ax is None:
        ax = plt.gca()
    if color is None:
        line, = ax.plot(sts.mean(), 0)
        color = line.get_color()
        line.remove()

    rates = []
    norm_bins = None
    bin_range = None
    
    for time_extent in event_time_ranges:
        rate, bins = peri_event_time_histogram(spike_times, time_extent,
                                               bin_size=bin_size)
        rates.append(rate)
        if norm_bins is None:
           time_range = time_extent[1] - time_extent[0]
           norm_bins = bins[:-1] - (time_extent[0] + (time_range / 2))
           bin_range = (-time_range/2, time_range/2)
        else:
           time_range = time_extent[1] - time_extent[0]
           bin_range = (min(bin_range[0], -time_range/2.), max(bin_range[1], time_range/2))
            
    rate_array = np.vstack(rates)
    avg_rate = np.mean(rate_array, axis=0)
    std_rate = np.std(rate_array, axis=0)

    ax.plot(norm_bins, avg_rate,
            color=color,
            linewidth=2)
    ax.fill_between(norm_bins, avg_rate-std_rate, avg_rate+std_rate,
                    alpha=0.1,
                    color=color,
                    linewidth=0)

    ax.axvline(0, color='black', linestyle=':')
    ax.set_xlim(bin_range)

    return norm_bins, rate_array

    
