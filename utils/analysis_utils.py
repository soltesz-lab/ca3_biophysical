import numpy as np
from collections import defaultdict 
from scipy.signal import spectrogram, hilbert, butter, lfilter, medfilt
from scipy.stats import pearsonr, spearmanr, zscore


#### circuit I/O

def transfer_circuit(circuitA, circuitB):
    for population_gid in circuitA.neurons.keys():
        if population_gid == 'Septal': continue
        population_infoA = circuitA.neurons[population_gid]
        population_infoB = circuitB.neurons[population_gid]
        for cell in population_infoA.keys():
            cell_infoA = population_infoA[cell]
            cell_infoB = population_infoB[cell]
            for (idx,(presynaptic_gid, nc, compartment)) in enumerate(cell_infoA.internal_netcons):
                presynapticB = cell_infoB.internal_netcons[idx]
                ncA = nc[0]
                ncB = presynapticB[1][0]
                try:
                    ncB.weight[0] = ncA.weight[0] + ncA.weight[1]
                except: 
                    ncB.weight[0] = ncA.weight[0]
            for external_id in cell_infoA.external_netcons.keys():
                external_cell_infoA = cell_infoA.external_netcons[external_id]
                external_cell_infoB = cell_infoB.external_netcons[external_id]
                for (idx,(presynaptic_gid, nc, compartment)) in enumerate(external_cell_infoA):
                    presynapticB = external_cell_infoB[idx]
                    ncA = nc[0]
                    ncB = presynapticB[1][0]
                    try: 
                        ncB.weight[0] = ncA.weight[0] + ncA.weight[1]
                    except: 
                        ncB.weight[0] = ncA.weight[0]
                            


###


def baks(spktimes, time, a=1.5, b=None):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)
    BAKS is a method for estimating firing rate from spike train data that uses kernel smoothing technique 
    with adaptive bandwidth determined using a Bayesian approach
    ---------------INPUT---------------
    - spktimes : spike event times [s]
    - time : time points at which the firing rate is estimated [s]
    - a : shape parameter (alpha) 
    - b : scale parameter (beta)
    ---------------OUTPUT---------------
    - rate : estimated firing rate [nTime x 1] (Hz)
    - h : adaptive bandwidth [nTime x 1]
    Based on "Estimation of neuronal firing rate using Bayesian adaptive kernel smoother (BAKS)"
    https://github.com/nurahmadi/BAKS
    """
    from scipy.special import gamma

    n = len(spktimes)
    sumnum = 0
    sumdenom = 0

    if b is None:
        b = 0.42
    b = float(n) ** b

    for i in range(n):
        numerator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a)
        denominator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a - 0.5)
        sumnum = sumnum + numerator
        sumdenom = sumdenom + denominator

    h = (gamma(a) / gamma(a + 0.5)) * (sumnum / sumdenom)
    rate = np.zeros((len(time),))
    for j in range(n):
        x = np.asarray(-((time - spktimes[j]) ** 2) / (2. * h ** 2), dtype=np.float128)
        K = (1. / (np.sqrt(2. * np.pi) * h)) * np.exp(x)
        rate = rate + K

    return rate, h

def saved_weight_change(connection_dict, weight_dict, src_id, dst_id):
    
    weight_before, weight_after = [], []

    this_weight_dict = weight_dict[dst_id]
    for dst_gid in connection_dict[dst_id].keys():
        if src_id not in connection_dict[dst_id][dst_gid]: continue
        src_gids = connection_dict[dst_id][dst_gid][src_id]
        has_updated_weights = len(this_weight_dict[dst_gid]) > len(src_gids)
        for i, src_gid in enumerate(src_gids):
            if has_updated_weights:
                weight_before.append(this_weight_dict[dst_gid][2*i])
                weight_after.append(this_weight_dict[dst_gid][2*i] + this_weight_dict[dst_gid][2*i + 1])
            else:
                weight_before.append(this_weight_dict[dst_gid][i])
                weight_after.append(this_weight_dict[dst_gid][i])
        
    pchange = []
    for (b,a) in zip(weight_before, weight_after):
        pchange.append((a-b)/(b+1.0e-9))
        
    return weight_before, weight_after, pchange
    


def external_weight_change(circuit, src_id, dst_id):
    weight_before, weight_after = [], []
    for key in circuit.neurons[dst_id].keys(): # exc population
        if src_id not in circuit.neurons[dst_id][key].external_netcons: continue
        for ncs in circuit.neurons[dst_id][key].external_netcons[src_id]: # receiving from MEC cells
            weight_before.append(ncs[1][0].weight[0])
            try:
                weight_after.append(ncs[1][0].weight[0] + ncs[1][0].weight[1])
            except:
                weight_after.append(ncs[1][0].weight[0])
        
    pchange = []
    for (b,a) in zip(weight_before, weight_after):
        pchange.append((a-b)/(b+1.0e-9))
        
    return weight_before, weight_after, pchange


def internal_weight_change(circuit, src_id, dst_id, valid_gids):
    weight_before, weight_after = [], []
    for key in circuit.neurons[dst_id].keys(): # exc population
        for (gid, nc, _) in circuit.neurons[dst_id][key].internal_netcons:
            if gid not in valid_gids: continue # presynaptic
            try:
                w, a = nc[0].weight[0], nc[0].weight[1]
                wa = w + a
                weight_before.append(nc[0].weight[0])
                weight_after.append(nc[0].weight[0] + nc[0].weight[1])
            except: 
                w = nc[0].weight[0]
                weight_before.append(w)
                weight_after.append(w)            
    pchange = []
    for (b,a) in zip(weight_before, weight_after):
        pchange.append((a-b)/(b+1.0e-9))
        
    return weight_before, weight_after, pchange

def restore_weights(diagram, file_path):
    pop_weights_dict = defaultdict(lambda: dict())
    saved_weights = { int(k): v for k,v in np.load(file_path).items() }
    for pop in diagram.wiring_information.keys():
        pop_id = diagram.pop2id[pop]
        ctype_offset = diagram.wiring_information[pop]['ctype offset']
        ncells = diagram.wiring_information[pop]['ncells']
        for gid in range(ctype_offset, ctype_offset+ncells):
            if gid in saved_weights:
                src_gids = saved_weights[gid]['src_gids']
                compartment_idxs = saved_weights[gid]['compartment_idxs']
                src_gid_weights = saved_weights[gid]['weights']
                src_gid_weights_upd = saved_weights[gid]['weights_upd']
                src_gid_dict = defaultdict(list)
                for src_pop in diagram.wiring_information.keys():
                    src_pop_id = diagram.pop2id[src_pop]
                    src_ctype_offset = diagram.wiring_information[src_pop]['ctype offset']
                    src_ncells = diagram.wiring_information[src_pop]['ncells']
                    this_src_gid_idxs = np.argwhere(np.logical_and(src_gids >= src_ctype_offset,
                                                                   src_gids < src_ctype_offset + src_ncells))[:,0]
                    src_gid_dict[src_pop_id] = (src_gids[this_src_gid_idxs],
                                                src_gid_weights[this_src_gid_idxs],
                                                src_gid_weights_upd[this_src_gid_idxs],
                                                compartment_idxs[this_src_gid_idxs])
                for src_pop in diagram.external_information.keys():
                    src_pop_id = diagram.external_pop2id[src_pop]
                    src_ctype_offset = diagram.external_information[src_pop]['ctype offset']
                    src_ncells = diagram.external_information[src_pop]['ncells']
                    this_src_gid_idxs = np.argwhere(np.logical_and(src_gids >= src_ctype_offset,
                                                                   src_gids < src_ctype_offset + src_ncells))[:,0]
                    src_gid_dict[src_pop_id] = (src_gids[this_src_gid_idxs],
                                                src_gid_weights[this_src_gid_idxs],
                                                src_gid_weights_upd[this_src_gid_idxs],
                                                compartment_idxs[this_src_gid_idxs])
                pop_weights_dict[pop_id][gid] = dict(src_gid_dict)
    return dict(pop_weights_dict)

def saved_weight_change(weights_dict, src_id, dst_id, valid_gids=None, valid_src_gids=None, return_gids=False):
    
    weight_before, weight_after = [], []
    output_src_gids, output_dst_gids = [], []

    this_weights_dict = weights_dict[dst_id]
    for dst_gid in this_weights_dict.keys():
        if valid_gids is not None and dst_gid not in valid_gids:
            continue
        if src_id not in this_weights_dict[dst_gid]: 
            continue
        src_gids, connection_weights, connection_weights_upd, _ = this_weights_dict[dst_gid][src_id]
        has_updated_weights = len(~np.isnan(connection_weights_upd)) > 0
        for i, src_gid in enumerate(src_gids):
            if (valid_src_gids is not None) and (src_gid not in valid_src_gids):
                continue
            if has_updated_weights and not np.isnan(connection_weights_upd[i]):
                weight_before.append(connection_weights[i])
                weight_after.append(connection_weights_upd[i])
            else:
                weight_before.append(connection_weights[i])
                weight_after.append(connection_weights[i])
            if return_gids:
                output_src_gids.append(src_gid)
                output_dst_gids.append(dst_gid)

    pchange = []
    for (b,a) in zip(weight_before, weight_after):
        pchange.append((a-b)/(b+1.0e-9))
        
    if return_gids:
        return weight_before, weight_after, pchange, output_src_gids, output_dst_gids
    else:
        return weight_before, weight_after, pchange


def scale_weights(weights_dict, src_id, dst_id, scale, valid_gids=None, valid_src_gids=None):

    assert scale >= 0.0
    if valid_src_gids is not None:
        valid_src_gids = np.fromiter(valid_src_gids, dtype=np.uint32)
    this_weights_dict = weights_dict[dst_id]
    for dst_gid in this_weights_dict.keys():
        if valid_gids is not None and dst_gid not in valid_gids:
            continue
        if src_id not in this_weights_dict[dst_gid]: 
            continue
        src_gids, connection_weights, connection_weights_upd, _ = this_weights_dict[dst_gid][src_id]
        mask = None
        print(f"gid {dst_gid} src_gids: {src_gids} valid_src_gids: {valid_src_gids}")
        if valid_src_gids is not None:
            mask = np.isin(src_gids, valid_src_gids)
        has_updated_weights = len(~np.isnan(connection_weights_upd)) > 0
        if mask:
            if has_updated_weights:
                w = connection_weights_upd[mask]
                connection_weights_upd[mask] = w*scale
            else:
                connection_weights[mask] *= scale
        else:
            if has_updated_weights:
                w = connection_weights_upd
                connection_weights_upd[:] = w*scale
            else:
                connection_weights[:] *= scale
            
    
def save_weights_dict(weights_dict, save_filepath):

    flat_weights_dict = {}

    for dst_pop_id in weights_dict:
        this_weights_dict = weights_dict[dst_pop_id]
        for dst_gid in this_weights_dict.keys():
            dst_gid_weights_dict = this_weights_dict[dst_gid]
            dst_gid_src_gids_list = []
            dst_gid_src_weights_list = []
            dst_gid_src_weights_upd_list = []
            dst_gid_compartment_idxs_list = []
            for src_pop_id in dst_gid_weights_dict:
                src_gids, src_gid_weights, src_gid_weights_upd, compartment_idxs = dst_gid_weights_dict[src_pop_id]
                dst_gid_src_gids_list.append(src_gids)
                dst_gid_src_weights_list.append(src_gid_weights)
                dst_gid_src_weights_upd_list.append(src_gid_weights_upd)
                dst_gid_compartment_idxs_list.append(compartment_idxs)
            dst_gid_compartment_idxs_array = np.concatenate(dst_gid_compartment_idxs_list, dtype=np.uint16)
            dst_gid_src_gids_array = np.concatenate(dst_gid_src_gids_list, dtype=np.uint32)
            dst_gid_src_weights_array = np.concatenate(dst_gid_src_weights_list, dtype=np.float32)
            dst_gid_src_weights_upd_array = np.concatenate(dst_gid_src_weights_upd_list, dtype=np.float32)
            flat_weights_dict[str(dst_gid)] = np.core.records.fromarrays((dst_gid_src_weights_array,
                                                                     dst_gid_src_weights_upd_array,
                                                                     dst_gid_src_gids_array,
                                                                     dst_gid_compartment_idxs_array),
                                                                    names='weights,weights_upd,src_gids,compartment_idxs')

                
    

    np.savez(save_filepath, **flat_weights_dict)

    
#####


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

####

def spike_bins(spike_times, start, finish, binsize=100, gids=None):
    
    if gids is None:
        gids = sorted(spike_times.keys())
        
    bins = np.arange(start, finish, binsize)
    nbins = len(bins)+1
    nspikes = np.zeros((nbins,),dtype=np.uint32)
    for gid in gids:
        sts = spike_times[gid]
        if (gids is not None) and (gid not in gids):
            continue
        sts = np.asarray(sts)
        this_ibins = np.digitize(sts, bins)
        for ibin in this_ibins:
            nspikes[ibin] += 1
            
    return nspikes

#### Ripple detection based on code from
#### https://github.com/edeno/Jadhav-2016-Data-Analysis/blob/master/src/ripple_detection.py

def array_shift(arr: np.ndarray, shift: int, fill: np.nan) -> np.ndarray:
    
    if shift == 0:
        return arr
    nas = None
    if fill is not None:
        nas = np.full(abs(shift), fill)
    else:
        nas = np.empty(abs(shift))    
    if shift > 0:
        res = arr[:-shift]
        return np.concatenate((nas,res))
    res = arr[-shift:]
    return np.concatenate((res,nas))


def get_series_start_end_times(time_index, series):
    '''Returns a two element tuple with of the start of the segment and the
     end of the segment. Each element is an numpy array, The input series
    must be a boolean ndarray where the index is time_index.
    '''
    is_start_time = (~array_shift(series, 1, False)) & series
    start_times = time_index[is_start_time]

    is_end_time = series & (~array_shift(series, -1, False))
    end_times = time_index[is_end_time]

    return start_times, end_times

def segment_boolean_series(time_index, series):
    '''Returns a list of tuples where each tuple contains the start time
     of segement and end time of segment. It takes a time index
     ndarray and boolean ndarray as input where the index is time_index.
    '''
    start_times, end_times = get_series_start_end_times(time_index, series)

    return [(start_time, end_time)
            for start_time, end_time in zip(start_times, end_times)
            if end_time > start_time]


def find_containing_interval(interval_candidates, target_interval):
    '''Returns the interval that contains the target interval out of a list
    of interval candidates.

    This is accomplished by finding the closest start time out of the
    candidate intervals, since we already know that one interval candidate
    contains the target interval (the segements above 0 contain the
    segments above the threshold)
    '''
    candidate_start_times = np.asarray(interval_candidates)[:, 0]
    closest_start_ind = np.max(
        (candidate_start_times - target_interval[0] <= 0).nonzero())
    return interval_candidates[closest_start_ind]


def expand_segment(segments_to_extend, containing_segments):
    '''Expand the boundaries of a segment if it is a subset of one of the
    containing segments.

    Parameters
    ----------
    segments_to_extend : list of 2-element tuples
        Elements are the start and end times
    containing_segments : list of 2-element tuples
        Elements are the start and end times

    Returns
    -------
    expanded_segments : list of 2-element tuples

    '''
    segments = [find_containing_interval(containing_segments, segment)
                for segment in segments_to_extend]
    return list(set(segments))  # remove duplicate segments


def expand_threshold_to_mean(time_index, is_above_mean, is_above_threshold):
    
    '''Expand segments above threshold to span range above mean.

    Parameters
    ----------
    is_above_mean : ndarray
        Time series indicator function specifying when the
        time series is above the mean. Index of the series is time.
    is_above_threshold : ndarray
        Time series indicator function specifying when the
        time series is above the the threshold. Index of the series is
        time.

    Returns
    -------
    extended_segments : list of 2-element tuples
        Elements correspond to the start and end time of segments

    '''
    above_mean_segments = segment_boolean_series(
        time_index, is_above_mean)
    above_threshold_segments = segment_boolean_series(
        time_index, is_above_threshold)
    return expand_segment(above_threshold_segments, above_mean_segments)


def merge_overlapping_ranges(ranges):
    '''Merge overlapping and adjacent ranges

    Parameters
    ----------
    ranges : iterable with 2-elements
        Element 1 is the start of the range.
        Element 2 is the end of the range.
    Yields
    -------
    sorted_merged_range : 2-element tuple
        Element 1 is the start of the merged range.
        Element 2 is the end of the merged range.

    >>> list(_merge_overlapping_ranges([(5,7), (3,5), (-1,3)]))
    [(-1, 7)]
    >>> list(_merge_overlapping_ranges([(5,6), (3,4), (1,2)]))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(_merge_overlapping_ranges([]))
    []

    References
    ----------
    .. [1] http://codereview.stackexchange.com/questions/21307/consolidate-
    list-of-ranges-that-overlap

    '''
    ranges = iter(sorted(ranges))
    current_start, current_stop = next(ranges)
    for start, stop in ranges:
        if start > current_stop:
            # Gap between segments: output current segment and start a new
            # one.
            yield current_start, current_stop
            current_start, current_stop = start, stop
        else:
            # Segments adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    yield current_start, current_stop

    
def get_ripple_candidates (spike_times, start, finish, binsize=100., z_threshold=1):

    spike_bins = np.arange(start, finish, binsize)
    sts_flat = np.concatenate(list(spike_times.values()))
    sts_chop = sts_flat[np.where( (sts_flat>=start) & (sts_flat<=finish)) [0]]
    nspikes, spike_edges = np.histogram(sts_chop, bins=spike_bins)

    time_index = spike_edges[:-1]
    z_nspikes = zscore(nspikes)

    is_above_threshold = np.asarray(z_nspikes >= z_threshold)
    is_above_mean = np.asarray(z_nspikes >= 0.)
    
    candidate_ripple_times = expand_threshold_to_mean(
        time_index, is_above_mean, is_above_threshold)
    return list(merge_overlapping_ranges(
        candidate_ripple_times))


def peri_event_time_histogram(spike_times,
                              time_extent,
                              bin_size=100,
                              normalized=False):
    ''' Returns the histogram of spike times
    
    Parameters:
        spike_times: An indicator array with the time of spikes
        time_extent: Tuple with the start and end times of the time period
                     under consideration
        bin_size: The size of the histogram time bins. Must be in the
                  same units as the time extent
        normalized: boolean where True means the firing rate is normalized
                    by its maximum firing rate
                     
    Returns:
        rate: The firing rate for each bin
        bins: The left edge of each histogram time bin
    '''
    sts_flat = np.concatenate(list(spike_times.values()))
    sts_chop = sts_flat[np.where( (sts_flat>=time_extent[0]) & (sts_flat<=time_extent[1])) [0]]
    
    number_of_bins = np.fix((time_extent[1] - time_extent[0]) / bin_size).astype(int)
    bin_count, bins = np.histogram(sts_chop, bins=number_of_bins, range=time_extent)
    zbin_count = zscore(bin_count)
        
    return zbin_count, bins
