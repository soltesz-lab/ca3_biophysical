import numpy as np
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
