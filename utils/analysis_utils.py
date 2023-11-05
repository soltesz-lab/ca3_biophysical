import numpy as np
from collections import defaultdict 
from scipy.signal import spectrogram, hilbert, butter, lfilter, medfilt


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
                
                src_gid_weights = saved_weights[gid]['weights']
                src_gid_weights_upd = saved_weights[gid]['weights_upd']
                src_gid_dict = defaultdict(list)
                for src_pop in diagram.wiring_information.keys():
                    src_pop_id = diagram.pop2id[src_pop]
                    src_ctype_offset = diagram.wiring_information[src_pop]['ctype offset']
                    src_ncells = diagram.wiring_information[src_pop]['ncells']
                    this_src_gid_idxs = np.argwhere(np.logical_and(src_gids >= src_ctype_offset,
                                                                   src_gids <= src_ctype_offset + src_ncells))[:,0]
                    
                    src_gid_dict[src_pop_id] = (src_gids[this_src_gid_idxs],
                                                src_gid_weights[this_src_gid_idxs],
                                                src_gid_weights_upd[this_src_gid_idxs])
                for src_pop in diagram.external_information.keys():
                    src_pop_id = diagram.external_pop2id[src_pop]
                    src_ctype_offset = diagram.external_information[src_pop]['ctype offset']
                    src_ncells = diagram.external_information[src_pop]['ncells']
                    this_src_gid_idxs = np.argwhere(np.logical_and(src_gids >= src_ctype_offset,
                                                                   src_gids <= src_ctype_offset + src_ncells))[:,0]
                    src_gid_dict[src_pop_id] = (src_gids[this_src_gid_idxs],
                                                src_gid_weights[this_src_gid_idxs],
                                                src_gid_weights_upd[this_src_gid_idxs])
                pop_weights_dict[pop_id][gid] = dict(src_gid_dict)
    return dict(pop_weights_dict)

def saved_weight_change(weights_dict, src_id, dst_id, valid_gids=None):
    
    weight_before, weight_after = [], []

    this_weights_dict = weights_dict[dst_id]
    for dst_gid in this_weights_dict.keys():
        if valid_gids is not None and dst_gid not in valid_gids:
            continue
        if src_id not in this_weights_dict[dst_gid]: 
            continue
        src_gids, connection_weights, connection_weights_upd = this_weights_dict[dst_gid][src_id]
        has_updated_weights = len(~np.isnan(connection_weights_upd)) > 0
        for i, src_gid in enumerate(src_gids):
            if has_updated_weights and not np.isnan(connection_weights_upd[i]):
                weight_before.append(connection_weights[i])
                weight_after.append(connection_weights_upd[i])
            else:
                weight_before.append(connection_weights[i])
                weight_after.append(connection_weights[i])
    pchange = []
    for (b,a) in zip(weight_before, weight_after):
        pchange.append((a-b)/(b+1.0e-9))
        
    return weight_before, weight_after, pchange

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
