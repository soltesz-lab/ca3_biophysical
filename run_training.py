#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from pprint import pprint
import sys, os
import pickle as pkl
from collections import defaultdict
import logging

from os.path import expanduser
home = expanduser("~")
model_home = os.path.join(home, 'src/model/ca3_biophysical/')
sys.path.append(os.path.join(home, 'model/ca3_biophysical/utils'))
sys.path.append(os.path.join(home, 'model/ca3_biophysical/cells'))
sys.path.append(os.path.join(home, 'bin/nrnpython3/lib/python'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from neuron import h
from neuron.units import ms, mV
h.nrnmpi_init()

from SetupConnections import *
from NeuronCircuit import Circuit
from analysis_utils import baks
from simtime import SimTimeEvent

print('constructing model..')
sys.stdout.flush()


delay = 500.
dt = 0.1

pc = h.ParallelContext()

params_path = os.path.join(model_home, 'params')
ar = Arena(os.path.join(params_path, 'arenaparams.yaml'))
ar.generate_population_firing_rates()
ar.generate_cue_firing_rates('LEC', 1.0)


cued = True

fr = ar.cell_information['LEC']['cell info'][0]['firing rate']

edge  = 12.5
lp    = 1

arena_size = ar.params['Arena']['arena size']
bin_size   = ar.params['Arena']['bin size']
mouse_speed = ar.params['Arena']['mouse speed']
nlaps       = ar.params['Arena']['lap information']['nlaps']

arena_map  = np.arange(0, 200,step=0.1)
cued_positions  = np.linspace(edge, 200-edge, nlaps*lp)
random_cue_locs = np.arange(len(cued_positions))

if pc.id() == 0:
    np.random.shuffle(random_cue_locs)
    logger.info(f"random_cue_locs = {random_cue_locs}")
random_cue_locs = pc.py_broadcast(random_cue_locs, 0)

time_for_single_lap = arena_size / mouse_speed * 1000.

frs_all = []
for i in range(nlaps):
    random_position = cued_positions[random_cue_locs[i]]
    to_roll = int( ( 100. - random_position) / 0.1 )
    fr_rolled = np.roll(fr, to_roll)
    frs_all.append(fr_rolled)

frs_all = np.asarray(frs_all)



# In[16]:


place_information = {'place ids': [0], 'place fracs': [0.80]}

diagram = WiringDiagram(os.path.join(params_path, 'circuitparams.yaml'), place_information)
diagram.generate_internal_connectivity()


place_ids = diagram.place_information[0]['place']
cue_ids = diagram.place_information[0]['not place']

external_kwargs = {}
external_kwargs['place information'] = diagram.place_information
external_kwargs['external place ids'] = [100, 101, 102]
external_kwargs['cue information'] = diagram.place_information
external_kwargs['external cue ids'] = [100, 101, 102]

diagram.generate_external_connectivity(ar.cell_information, **external_kwargs)
diagram.generate_septal_connectivity()


print('generating spike times..')
sys.stdout.flush()
ar.generate_spike_times('MF', dt=dt, delay=delay)
ar.generate_spike_times('MEC', dt=dt, delay=delay)
ar.generate_spike_times('LEC', dt=dt, delay=delay, cued=cued)
ar.generate_spike_times('Background', dt=dt, delay=delay)
print('generated spike times..')
sys.stdout.flush()


# In[12]:


def pull_spike_times(population2info_dict):
    spike_times = {}
    gids = np.sort(list(population2info_dict.keys()))
    for gid in gids:
        gid_info = population2info_dict[gid]
        if 'spike times' in gid_info:
            spike_times[gid] = gid_info['spike times']
    return spike_times

mf_spike_times  = pull_spike_times(ar.cell_information['MF']['cell info'])
mec_spike_times = pull_spike_times(ar.cell_information['MEC']['cell info'])
lec_spike_times = pull_spike_times(ar.cell_information['LEC']['cell info'])
bk_spike_times  = pull_spike_times(ar.cell_information['Background']['cell info'])


print('constructing circuit..')
sys.stdout.flush()

circuit = Circuit(params_prefix=params_path, 
                  params_filename='circuitparams.yaml',
                  arena_params_filename='arenaparams.yaml', 
                  internal_pop2id=diagram.pop2id, 
                  external_pop2id=diagram.external_pop2id, 
                  external_spike_times = {100: mf_spike_times,
                                          101: mec_spike_times,
                                          102: lec_spike_times,
                                          103: bk_spike_times})
print('building cells..')
sys.stdout.flush()
circuit.build_cells()

circuit.build_internal_netcons(diagram.internal_adj_matrices, diagram.internal_ws)
circuit.build_external_netcons(100, diagram.external_adj_matrices[100], diagram.external_ws[100])
circuit.build_external_netcons(101, diagram.external_adj_matrices[101], diagram.external_ws[101])
circuit.build_external_netcons(102, diagram.external_adj_matrices[102], diagram.external_ws[102])
circuit.build_external_netcons(103, diagram.external_adj_matrices[103], diagram.external_ws[103])
#circuit.record_lfp([0,1])
#circuit.build_septal_netcons(diagram.septal_adj_matrices)



import time


def get_cell_population_spikes(c,pop_id):
    spike_times_dict = c.get_cell_spikes(pop_id)
    return spike_times_dict

def get_ext_population_spikes(c,pop_id):
    spike_vec_dict = defaultdict(list)
    spike_times_vec = c.external_spike_time_recs[pop_id]
    spike_gids_vec  = c.external_spike_gid_recs[pop_id]
    for spike_t, spike_gid in zip(spike_times_vec, spike_gids_vec):
        spike_gid = int(spike_gid)
        spike_vec_dict[spike_gid].append(spike_t)
    return spike_vec_dict


def get_population_voltages(c,pop_id,rec_dt=0.05):
    v_vec_dict = {}
    for cid, cell in c.neurons[pop_id].items():
        v_vec = h.Vector()
        try:
            v_vec.record(cell.axon(0.5)._ref_v, rec_dt)
        except:
            v_vec.record(cell.soma(0.5)._ref_v, rec_dt)
        gid = c.ctype_offsets[pop_id] + cid
        v_vec_dict[gid] = v_vec
    return v_vec_dict

def save_connections(pc, circ, save_filepath):
    complete_connections = {}
    for population_id in circ.neurons.keys():
        if population_id == 'Septal': continue
        population_info = circ.neurons[population_id]
        for cell_gid in population_info.keys():
            cell_info_to_save = []
            cell_info = population_info[cell_gid]
            if not hasattr(cell_info, 'internal_netcons'):
                continue
            for (src_gid, nc, _) in cell_info.internal_netcons:
                for netcon in nc:
                    assert src_gid == int(netcon.srcgid())
                    cell_info_to_save.append(netcon.srcgid())
            for external_id in cell_info.external_netcons.keys():
                external_cell_info = cell_info.external_netcons[external_id]
                for (src_gid, nc, compartment) in external_cell_info:
                    for netcon in nc:
                        cell_info_to_save.append(netcon.srcgid())
            complete_connections[str(cell_gid)] = cell_info_to_save
    all_complete_connections = pc.py_gather(complete_connections, 0)

    if pc.id() == 0:
        complete_connections = {}
        for d in all_complete_connections:
            complete_connections.update(d)
        np.savez(save_filepath, **complete_connections)
        
    pc.barrier()

def save_v_vecs(pc, save_filepath, v_vecs):

    v_vec_dict = { k: np.asarray(v, dtype=np.float32) for k,v in v_vecs.items() }
    all_v_vecs = pc.py_gather(v_vec_dict, 0)

    if pc.id() == 0:
        v_vecs = {}
        for d in all_v_vecs:
            v_vecs.update([(str(k),v) for (k,v) in d.items()])
        np.savez(save_filepath, **v_vecs)

    pc.barrier()

def save_spike_vecs(pc, save_filepath, *spike_time_dicts):

    all_spike_dicts = pc.py_gather(spike_time_dicts, 0)

    if pc.id() == 0:
        spike_dict = {}
        for ds in all_spike_dicts:
            for d in ds:
                spike_dict.update([(str(k),v) for (k,v) in d.items()])
        np.savez(save_filepath, **spike_dict)

    pc.barrier()

pc = circuit.pc

save_connections(pc, circuit, f"data/0801-cue-ee-ei-connections.npz")

exc_v_vecs     = get_population_voltages(circuit, 0)
#pvbc_v_vecs    = get_population_voltages(circuit, 1)
# aac_v_vecs   = get_population_voltages(2)
# bis_v_vecs   = get_population_voltages(3)
# olm_v_vecs   = get_population_voltages(4)
# isccr_v_vecs = get_population_voltages(5)
# iscck_v_vecs = get_population_voltages(6)


t_vec = h.Vector()  # Time stamp vector
t_vec.record(h._ref_t)

tic = time.time()

h.dt = 0.025
h.celsius = 37.
h.tstop =  time_for_single_lap * nlaps + 500

if pc.id() == 0:
    print(f'starting simulation for {nlaps} lap(s) until {h.tstop} ms..')
    sys.stdout.flush()
    
simtime = SimTimeEvent(pc, h.tstop, 8.0, 10, 0)

pc.set_maxstep(10 * ms)

t = h.Vector().record(h._ref_t)
h.finitialize(-65 * mV)
pc.psolve(h.tstop * ms)

elapsed = time.time() - tic
pc.barrier()

if pc.id() == 0:
    print('simulation took %0.3f seconds' % elapsed)

def save_connection_weights(pc, circ, save_filepath):
    complete_weights = {}
    for population_id in circ.neurons.keys():
        if population_id == 'Septal': continue
        population_info = circ.neurons[population_id]
        for cell_gid in population_info.keys():
            connection_weights = []
            connection_weights_upd = []
            src_gids = []
            cell_info = population_info[cell_gid]
            if not hasattr(cell_info, 'internal_netcons'):
                continue
            for (presynaptic_id, nc, _) in cell_info.internal_netcons:
                for netcon in nc:
                    src_gids.append(int(netcon.srcgid()))
                    connection_weights.append(netcon.weight[0])
                    if len(netcon.weight) == 3:
                        connection_weights_upd.append(netcon.weight[1])
                    else:
                        connection_weights_upd.append(0.0)
            for external_id in cell_info.external_netcons.keys():
                external_cell_info = cell_info.external_netcons[external_id]
                for (idx,(presynaptic_gid, nc, compartment)) in enumerate(external_cell_info):
                    for netcon in nc:
                        src_gids.append(int(netcon.srcgid()))
                        connection_weights.append(netcon.weight[0])
                        if len(netcon.weight) == 3: 
                            connection_weights_upd.append(netcon.weight[1])
                        else:
                            connection_weights_upd.append(0.0)
            connection_info = np.core.records.fromarrays((np.asarray(connection_weights, dtype=np.float32),
                                                          np.asarray(connection_weights_upd, dtype=np.float32),
                                                          np.asarray(src_gids, dtype=np.uint32)),
                                                         names='weights,weights_upd,src_gids')
            complete_weights[str(cell_gid)] = connection_info

    all_complete_weights = pc.py_gather(complete_weights, 0)

    if pc.id() == 0:
        complete_weights = {}
        for d in all_complete_weights:
            complete_weights.update(d)
        np.savez(save_filepath, **complete_weights)
        
    pc.barrier()

    

ext_spikes_MF   = get_ext_population_spikes(circuit, 100)
ext_spikes_MEC  = get_ext_population_spikes(circuit, 101)
ext_spikes_LEC  = get_ext_population_spikes(circuit, 102)
ext_spikes_Bk   = get_ext_population_spikes(circuit, 103)

save_spike_vecs(pc, f"data/ext_spikes_0801-cue-ee-ei-nlaps-{nlaps}",
                ext_spikes_MF,
                ext_spikes_MEC,
                ext_spikes_LEC,
                ext_spikes_Bk)

cell_spikes_PC    = get_cell_population_spikes(circuit,0)
cell_spikes_PVBC  = get_cell_population_spikes(circuit,1)

save_spike_vecs(pc, f"data/cell_spikes_0801-cue-ee-ei-nlaps-{nlaps}",
                cell_spikes_PC,
                cell_spikes_PVBC)
                
        
save_connections(pc, circuit, f"data/0801-cue-ee-ei-connections.npz")

save_connection_weights(pc, circuit, f"params/0801-cue-ee-ei-nlaps-{nlaps}-dt-zerodot1-scale-2-v1.npz")

save_v_vecs(pc, f"data/v_vecs_0801-cue-ee-ei-nlaps-{nlaps}", exc_v_vecs)

pc.runworker()
pc.done()
h.quit()
