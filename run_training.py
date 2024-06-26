#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from pprint import pprint
import sys, os
import pickle as pkl
from collections import defaultdict
import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from neuron import h
from neuron.units import ms, mV
h.nrnmpi_init()

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


def get_population_voltages(c,pop_id,rec_dt=0.1):
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


def pull_spike_times(population2info_dict):
    spike_times = {}
    gids = np.sort(list(population2info_dict.keys()))
    for gid in gids:
        gid_info = population2info_dict[gid]
        if 'spike times' in gid_info:
            spike_times[gid] = gid_info['spike times']
    return spike_times

def main():
    parser = argparse.ArgumentParser(
        description="Run CA3 cue cell training."
    )

    parser.add_argument(
        "--circuit-config",
        required=True,
        type=str,
        help="Name of circuit configuration file. ",
    )

    parser.add_argument(
        "--arena-config",
        required=True,
        type=str,
        help="Name of arena configuration file. ",
    )

    parser.add_argument(
        "--model-home",
        required=False,
        type=str,
        help="Path to model home directory. ",
    )

    parser.add_argument(
        "--data-prefix",
        required=False,
        default="data",
        type=str,
        help="Path to data files. ",
    )

    parser.add_argument(
        "-c", "--config-id",
        required=True,
        type=str,
        help="Configuration identifier. ",
    )

    parser.add_argument(
        "-s", "--save-weights-every",
        required=False,
        type=int, default=-1,
        help="Save weights every n laps. ",
    )

    args = parser.parse_args()

    model_home = args.model_home
    sys.path.append(os.path.join(model_home, 'utils'))
    sys.path.append(os.path.join(model_home, 'cells'))

    circuit_config_file = args.circuit_config
    arena_config_file = args.arena_config

    config_id = args.config_id

    data_prefix = args.data_prefix
    save_weights_every = args.save_weights_every
    
    from SetupConnections import WiringDiagram, Arena
    from NeuronCircuit import Circuit, save_v_vecs, save_netcon_data, save_spike_vecs
    from simtime import SimTimeEvent

    print('constructing model..')
    sys.stdout.flush()

    delay = 500.
    dt = 0.1

    pc = h.ParallelContext()
    
    params_path = os.path.join(model_home, 'params')
    arena_config_path = os.path.join(params_path, arena_config_file)
    circuit_config_path = os.path.join(params_path, circuit_config_file)

    
    ar = Arena(os.path.join(params_path, arena_config_path))
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
    random_cue_locs = pc.py_broadcast(random_cue_locs, 0)
        
    time_for_single_lap = arena_size / mouse_speed * 1000.
    
    frs_all = []
    for i in range(nlaps):
        random_position = cued_positions[random_cue_locs[i]]
        to_roll = int( ( 100. - random_position) / 0.1 )
        fr_rolled = np.roll(fr, to_roll)
        frs_all.append(fr_rolled)

    frs_all = np.asarray(frs_all)

    place_information = {'place ids': [0], 'place fracs': [0.80]}
    
    diagram = WiringDiagram(circuit_config_path, place_information)

    place_ids = diagram.place_information[0]['place']
    cue_ids = diagram.place_information[0]['not place']
    
    internal_kwargs = {}
    internal_kwargs['place information'] = diagram.place_information
    internal_kwargs['cue information'] = diagram.place_information
    
    diagram.generate_internal_connectivity(**internal_kwargs)

    
    external_kwargs = {}
    external_kwargs['place information'] = diagram.place_information
    external_kwargs['external place ids'] = [100, 101, 102]
    external_kwargs['cue information'] = diagram.place_information
    external_kwargs['external cue ids'] = [100, 101, 102]

    diagram.generate_external_connectivity(ar.cell_information, **external_kwargs)
    diagram.generate_septal_connectivity()
    
    
    ar.generate_spike_times('MF', dt=dt, delay=delay)
    ar.generate_spike_times('MEC', dt=dt, delay=delay)
    ar.generate_spike_times('LEC', dt=dt, delay=delay, cued=cued)
    ar.generate_spike_times('Background', dt=dt, delay=delay)
    sys.stdout.flush()


    mf_spike_times  = pull_spike_times(ar.cell_information['MF']['cell info'])
    mec_spike_times = pull_spike_times(ar.cell_information['MEC']['cell info'])
    lec_spike_times = pull_spike_times(ar.cell_information['LEC']['cell info'])
    bk_spike_times  = pull_spike_times(ar.cell_information['Background']['cell info'])

    print('constructing circuit..')
    sys.stdout.flush()
    
    circuit = Circuit(params_prefix=params_path, 
                      params_filename=circuit_config_file,
                      arena_params_filename=arena_config_file,
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

    pc = circuit.pc
    
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

    if save_weights_every < 0:
        save_weights_every = nlaps
        
    t = h.Vector().record(h._ref_t)
    simtime = SimTimeEvent(pc, time_for_single_lap * nlaps + delay, 8.0, 10, 0)
    mindelay = pc.set_maxstep(10 * ms)
    h.finitialize(-65 * mV)

    for ilap in range(nlaps):
        
        h.tstop = time_for_single_lap*(ilap + 1) + delay

        if pc.id() == 0:
            print(f'starting simulation for lap {ilap}/{nlaps} until {h.tstop} ms..')
            sys.stdout.flush()
            
        pc.set_maxstep(10 * ms)
        pc.psolve(h.tstop - mindelay)
    
        elapsed = time.time() - tic
        pc.barrier()

        if (ilap + 1) % save_weights_every == 0:
            save_netcon_data(pc, circuit, 
                             os.path.join(data_prefix, f"{config_id}-cue-ee-ei-nlaps-{(ilap+1)}-dt-zerodot1-scale-2-v1.npz"))
    
        if pc.id() == 0:
            print('simulation took %0.3f seconds' % elapsed)

    

    ext_spikes_MF   = get_ext_population_spikes(circuit, 100)
    ext_spikes_MEC  = get_ext_population_spikes(circuit, 101)
    ext_spikes_LEC  = get_ext_population_spikes(circuit, 102)
    ext_spikes_Bk   = get_ext_population_spikes(circuit, 103)

    save_spike_vecs(pc, 
                    os.path.join(data_prefix, f"ext_spikes_{config_id}-cue-ee-ei-nlaps-{nlaps}"),
                    ext_spikes_MF,
                    ext_spikes_MEC,
                    ext_spikes_LEC,
                    ext_spikes_Bk)

    cell_spikes_PC    = get_cell_population_spikes(circuit,0)
    cell_spikes_PVBC  = get_cell_population_spikes(circuit,1)

    save_spike_vecs(pc, 
                    os.path.join(data_prefix, f"cell_spikes_{config_id}-cue-ee-ei-nlaps-{nlaps}"),
                    cell_spikes_PC,
                    cell_spikes_PVBC)
                
        
    
    save_v_vecs(pc, 
                os.path.join(data_prefix, f"v_vecs_{config_id}-cue-ee-ei-nlaps-{nlaps}"), 
                exc_v_vecs)
    
    pc.runworker()
    pc.done()
    h.quit()
    

if __name__ == "__main__":
    main()
