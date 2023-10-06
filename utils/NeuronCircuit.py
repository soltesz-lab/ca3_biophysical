
import numpy as np
import copy
import yaml
import os
import sys
import pickle
import logging

from ca3pyr_dhh import ca3pyrcell
from pvbc import PVBC
from axoaxonic import AAC
from bis import BiS
from olm import OLM
from isccr import ISCCR
from iscck import ISCCK

from pvbc_v2 import PVBC_v2
from axoaxonic_v2 import AAC_v2
from bis_v2 import BiS_v2
from olm_v2 import OLM_v2

from septal import Septal
from neuron import h
from ca3_neuron_utils import create_netcon

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class Circuit(object):
    
    def __init__(self, arena_params_filename, params_filename, internal_pop2id, external_pop2id, external_spike_times, params_prefix='.'):
        self.pc = h.ParallelContext()

        self.params_prefix = params_prefix
        self.cell_params = None
        self.arena_cells = None
        self._read_params(params_prefix, params_filename, arena_params_filename)
        self.cell_params_path = os.path.join(params_prefix, params_filename)
        
        self.internal_pop2id = internal_pop2id
        self.external_pop2id = external_pop2id

        self.ctype_offsets = {}
        self.neurons  = {}
        self.external_spike_times = {}
        self.external_spike_time_recs = {}
        self.external_spike_gid_recs = {}
        
        self.lfp = None

        self.external_spike_times = external_spike_times

    def _read_params(self, params_prefix, params_filename, arena_params_filename):
        if self.pc.id() == 0:
            with open(os.path.join(params_prefix, params_filename), 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                self.cell_params = fparams['Circuit']
            with open(os.path.join(params_prefix, arena_params_filename), 'r') as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)           
                self.arena_cells = fparams['Arena Cells']
        self.pc.barrier()
        self.cell_params = self.pc.py_broadcast(self.cell_params, 0)
        self.arena_cells = self.pc.py_broadcast(self.arena_cells, 0)
            
            
    def build_cells(self):
        ext_types    = list(self.arena_cells.keys())
        cell_types   = self.cell_params['cells']
        cell_numbers = self.cell_params['ncells']
        ctype_offset = 0

        ca3pyr_props = None
        if self.pc.id() == 0:
            ca3pyr_props_path = os.path.join(self.params_prefix, 'CA3_Bakerb_marascoProp.pickle')
            with open(ca3pyr_props_path, "rb") as f:
                ca3pyr_props = pickle.load(f, encoding='latin1')
        self.pc.barrier()
        ca3pyr_props = self.pc.py_broadcast(ca3pyr_props, 0)
        
        for (i,ctype) in enumerate(cell_types):
            cidx = self.internal_pop2id[ctype]
            ncells = cell_numbers[i]
            self.ctype_offsets[cidx] = ctype_offset
            self.neurons[cidx] = {}
            for ctype_id in range(int(self.pc.id()), ncells, int(self.pc.nhost())):
                gid = ctype_id + ctype_offset
                if ctype == 'ca3pyr':
                     cell = ca3pyrcell(gid, 'B', ca3pyr_props)
                elif ctype == 'pvbc':
                    cell = PVBC(gid)
                elif ctype == 'axoaxonic':
                    cell = AAC(gid)
                elif ctype == 'cckbc':
                    cell = CCKBC_v2(gid)
                elif ctype == 'bis':
                    cell = BiS(gid)
                elif ctype == 'olm':
                    cell = OLM_v2(gid)
                elif ctype == 'isccr':
                    cell = ISCCR(gid)
                elif ctype == 'iscck':
                    cell = ISCCK(gid)
                else:
                    raise RuntimeError(f"Unknown cell type {ctype}")
                
                self.neurons[cidx][ctype_id] = cell

                
                ## Tell the ParallelContext that this cell is
                ## a source for all other hosts. NetCon is temporary.
                self.pc.set_gid2node(gid, self.pc.id())
                self.pc.cell(gid, cell.spike_detector) # Associate the cell with this host and gid
                
            ctype_offset += ncells

        for (i,ctype) in enumerate(ext_types):
            cidx = self.external_pop2id[ctype]
            ncells = self.arena_cells[ctype]['ncells']
            self.ctype_offsets[cidx] = ctype_offset
            self.neurons[cidx] = {}
            self.external_spike_time_recs[cidx] = h.Vector()
            self.external_spike_gid_recs[cidx] = h.Vector()
            for ctype_id in range(int(self.pc.id()), ncells, int(self.pc.nhost())):
                gid = ctype_id + ctype_offset
                if ctype == 'MF':
                    stimcell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stimcell.play(vec)
                elif ctype == 'MEC':
                    stimcell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stimcell.play(vec)
                elif ctype == 'LEC':
                    stimcell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stimcell.play(vec)
                elif ctype == 'Background':
                    stimcell = h.VecStim()
                    vec = h.Vector(self.external_spike_times[cidx][ctype_id])
                    stimcell.play(vec)
                else:
                    raise RuntimeError(f"Unknown cell type {ctype}")
                self.neurons[cidx][ctype_id] = stimcell
                self.pc.set_gid2node(gid, self.pc.id())
                spike_detector = h.NetCon(stimcell, None)
                self.pc.cell(gid, spike_detector) # Associate the cell with this host and gid
                self.pc.spike_record(gid,
                                     self.external_spike_time_recs[cidx],
                                     self.external_spike_gid_recs[cidx])
            ctype_offset += ncells
                
                
        if 'Septal' in self.cell_params.keys():
            ncells = self.cell_params['Septal']['ncells']
            self.neurons['Septal'] = {}
            for ctype_id in range(ncells):
                gid = ctype_id + ctype_offset
                sc = Septal(gid, self.cell_params['Septal']['parameters'])
                self.neurons['Septal'][ctype_id] = sc
                self.pc.set_gid2node(gid, self.pc.id())
                self.pc.cell(gid, sc.spike_detector) # Associate the cell with this host and gid
                
            ctype_offset += ncells
                
    def build_internal_netcons(self, internal_adj_matrices, internal_ws, seed=1e6):
        rnd = np.random.RandomState(seed=int(seed))
        src_population_ids = list(internal_adj_matrices.keys())
        for src_pop_id in src_population_ids:
            src_offset = self.ctype_offsets[src_pop_id]
            dst_population_ids = list(internal_adj_matrices[src_pop_id].keys())
            for dst_pop_id in dst_population_ids:
                adj_matrix = internal_adj_matrices[src_pop_id][dst_pop_id]
                dst_neurons = self.neurons[dst_pop_id] # PYR to PVBC, for example
                this_internal_ws = internal_ws[src_pop_id][dst_pop_id]
                synapse_information = self.cell_params['internal connectivity'][src_pop_id][dst_pop_id]['synapse']
                for i in range(adj_matrix.shape[0]):
                    src_gid = i + src_offset
                    for j in range(adj_matrix.shape[1]):
                        nconnections = adj_matrix[i,j]
                        if nconnections == 0: continue

                        if not ((j in dst_neurons) and self.pc.gid_exists(dst_neurons[j].gid)):
                            continue

                        ws               = this_internal_ws[(i,j)]
                        compartments     = synapse_information['compartments']
                        rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                        chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]
                        
                        for con_num in range(nconnections):
                            compartment = chosen_compartments[con_num]
                            compartment_idx = rnd_compartments[con_num]
                            ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[j], 
                                                synapse_information, compartment, self.cell_params,
                                                weight_scale=ws)

                            dst_neurons[j].internal_netcons.append( (src_gid, ncs, compartment, compartment_idx) )


    def build_external_netcons(self, src_pop_id, external_adj_matrices, external_ws, seed=1e7):
        seed = int(seed) + src_pop_id
        rnd  = np.random.RandomState(seed=seed)
        
        src_offset = self.ctype_offsets[src_pop_id]
        dst_population_ids = list(external_adj_matrices.keys())
        for dst_pop_id in dst_population_ids:
            adj_matrix = external_adj_matrices[dst_pop_id]
            this_external_ws = external_ws[dst_pop_id]
            dst_neurons = self.neurons[dst_pop_id] # PYR to PVBC, for example
            synapse_information = self.cell_params['external connectivity'][src_pop_id][dst_pop_id]['synapse']
            for i in range(adj_matrix.shape[0]):
                src_gid = i + src_offset
                for j in range(adj_matrix.shape[1]):
                    nconnections = adj_matrix[i,j]
                    if nconnections == 0: continue

                    if not ((j in dst_neurons) and self.pc.gid_exists(dst_neurons[j].gid)):
                        continue
                    
                    ws               = this_external_ws[(i,j)]
                    compartments     = synapse_information['compartments']
                    rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                    chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]
                    
                    for con_num in range(nconnections):
                        compartment_idx = rnd_compartments[con_num]
                        compartment = chosen_compartments[con_num]
                        ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[j], 
                                            synapse_information, compartment, self.cell_params,
                                            weight_scale=ws)
                        if src_pop_id not in self.neurons[dst_pop_id][j].external_netcons: 
                            self.neurons[dst_pop_id][j].external_netcons[src_pop_id] = []
                        self.neurons[dst_pop_id][j].external_netcons[src_pop_id].append( (src_gid, ncs, compartment, compartment_idx) )
                        
    def build_septal_netcons(self, src_pop_id, septal_adj_matrices, seed=1e8):
        rnd = np.random.RandomState(seed=int(seed))
        src_offset = self.ctype_offsets[src_pop_id]
        dst_population_ids = list(septal_adj_matrices.keys())
        for dst_pop_id in dst_population_ids:
            adj_matrix = septal_adj_matrices[dst_pop_id]
            
            synapse_information = self.cell_params['Septal']['connectivity'][dst_pop_id]['synapse']
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    nconnections = adj_matrix[i,j]
                    if nconnections == 0: continue
                    compartments     = synapse_information['compartments']
                    rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                    chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]

                    for con_num in range(nconnections):
                        compartment_idx = rnd_compartments[con_num]
                        compartment = chosen_compartments[con_num]
                        # TODO: provide source gids
                        ncs = create_netcon(self.pc, 'Septal', dst_pop_id,
                                            self.neurons['Septal'][i],
                                            self.neurons[dst_pop_id][j], 
                                            synapse_information, compartment, self.cell_params)
                        self.neurons[dst_pop_id][j].internal_netcons.append( (self.neurons['Septal'][i].gid, ncs,
                                                                              compartment, compartment_idx) )

    def load_netcons(self, connections_dict, src_population_ids, dst_population_ids, connectivity_type='internal connectivity'):

        rank = int(self.pc.id())
        cell_numbers_dict = {}
        if connectivity_type == 'internal connectivity':
            cell_types = self.cell_params['cells']
            for i, ctype in enumerate(cell_types):
                cidx = self.internal_pop2id[ctype]
                ncells = self.cell_params['ncells'][i]
                cell_numbers_dict[cidx] = ncells
        elif connectivity_type == 'external connectivity':
            ext_types = list(self.arena_cells.keys())
            for ctype in ext_types:
                cidx = self.external_pop2id[ctype]
                ncells = self.arena_cells[ctype]['ncells']
                cell_numbers_dict[cidx] = ncells
        else:
            raise RuntimeError(f"unknown connectivity type {connectivity_type}")
            

        
        for dst_pop_id in dst_population_ids:
            dst_neurons = self.neurons[dst_pop_id]
            for i in dst_neurons:
                netcon_count = 0
                if not ((i in dst_neurons) and self.pc.gid_exists(dst_neurons[i].gid)):
                    continue
                dst_gid = dst_neurons[i].gid
                connection_info = connections_dict[dst_gid]
                src_gids = connection_info['src_gids']
                comp_idxs = connection_info['compartment_idxs']
                src_weights = connection_info['weights']
                src_weights_upd = connection_info['weights_upd']
                for src_pop_id in src_population_ids:
                    if not dst_pop_id in self.cell_params[connectivity_type][src_pop_id]:
                        continue
                    synapse_information = self.cell_params[connectivity_type][src_pop_id][dst_pop_id]['synapse']
                    src_offset = self.ctype_offsets[src_pop_id]
                    src_ncells = cell_numbers_dict[src_pop_id]
                    this_src_idxs = np.asarray(np.argwhere(np.logical_and(src_gids >= src_offset,
                                                                          src_gids < src_offset + src_ncells)).flat)
                    this_src_gids = src_gids[this_src_idxs]
                    this_src_comp_idxs = comp_idxs[this_src_idxs]
                    this_src_weights = src_weights[this_src_idxs]
                    this_src_weights_upd = src_weights_upd[this_src_idxs]
                    
                    for src_gid, comp_idx, src_weight, src_weight_upd in zip(this_src_gids,
                                                                             this_src_comp_idxs,
                                                                             this_src_weights,
                                                                             this_src_weights_upd):
                        compartments = synapse_information['compartments']
                        compartment  = compartments[comp_idx]
                        ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[i], 
                                            synapse_information, compartment, self.cell_params,
                                            weight0=src_weight, weight_upd=src_weight_upd)
                        netcon_count += len(ncs)
                        if connectivity_type == "internal connectivity":
                            dst_neurons[i].internal_netcons.append( (src_gid, ncs, compartment, comp_idx) )
                        elif connectivity_type == "external connectivity":
                            if src_pop_id not in dst_neurons[i].external_netcons: 
                                dst_neurons[i].external_netcons[src_pop_id] = []
                            dst_neurons[i].external_netcons[src_pop_id].append( (src_gid, ncs, compartment, comp_idx) )
                        else:
                            raise RuntimeError(f"unknown connectivity type {connectivity_type}")
                if int(self.pc.id()) == 0:
                    logger.info(f"gid {dst_gid}: created {netcon_count} netcons")
                            
    def get_cell_spikes(self, group_id):
        neurons = self.neurons[group_id]
        spike_times = {}
        for k in list(neurons.keys()):
            cell = neurons[k]
            spike_times[cell.gid] = list(cell.spike_times)
        return spike_times
    
    def record_lfp(self, population_ids):
        neurons = self.neurons
        self.lfp = []
        for pop_id in population_ids:
            current_neural_population = neurons[pop_id]
            for gid in current_neural_population.keys():
                current_neuron = current_neural_population[gid]
                for syntype in current_neuron.synGroups:
                    for synlocation in current_neuron.synGroups[syntype]:
                        if synlocation == 'soma' or synlocation == 'oriensProximal' or synlocation == 'lucidum':
                            synapses = current_neuron.synGroups[syntype][synlocation]
                            for pid in synapses.keys():
                                syns = synapses[pid]
                                for syn in syns:
                                    curr = h.Vector()
                                    curr.record(syn._ref_i)
                                    self.lfp.append(curr)
                    
        
def restore_netcons(pc, circ, input_filepath, root=0):

    connections_dict = None
    rank = int(pc.id())
    nhost = int(pc.nhost())
    if rank == root:
        connections_data = np.load(input_filepath)
        connections_dict = {int(k): v for k,v in connections_data.items()}
        
    src_gids_list = []
    pc.barrier()
    for population_id in circ.neurons.keys():
        if population_id == 'Septal': continue
        ctype_offset = circ.ctype_offsets[population_id]
        population_info = circ.neurons[population_id]
        src_gids_list.append(np.asarray(list(population_info.keys())) + ctype_offset)
    src_gids = np.concatenate(src_gids_list)
    pc.barrier()
    src_gids_per_rank = pc.py_gather(src_gids, root)
        
    src_data = [{idx: connections_dict[idx] for idx in src_gids_per_rank[i] if idx in connections_dict}
                for i in range(nhost)] if rank == root else None
    connections_dict = pc.py_scatter(src_data, root)

    dst_population_ids = list([circ.internal_pop2id[pop] for pop in circ.cell_params['cells']])
    src_population_ids = list([circ.internal_pop2id[pop] for pop in circ.cell_params['cells']])
    circ.load_netcons(connections_dict, src_population_ids, dst_population_ids, connectivity_type='internal connectivity')
    src_population_ids = list([circ.external_pop2id[pop] for pop in list(circ.arena_cells.keys())])
    circ.load_netcons(connections_dict, src_population_ids, dst_population_ids, connectivity_type='external connectivity')


def save_netcon_data(pc, circ, save_filepath):
    complete_weights = {}
    for population_id in circ.neurons.keys():
        if population_id == 'Septal': continue
        ctype_offset = circ.ctype_offsets[population_id]
        population_info = circ.neurons[population_id]
        for cell_id in population_info.keys():
            cell_gid = ctype_offset + int(cell_id)
            connection_weights = []
            connection_weights_upd = []
            src_gids = []
            compartment_idxs = []
            cell_info = population_info[cell_id]
            if not hasattr(cell_info, 'internal_netcons'):
                continue
            for (presynaptic_id, ncs, _, compartment_idx) in cell_info.internal_netcons:
                for netcon in ncs:
                    compartment_idxs.append(compartment_idx)
                    src_gids.append(int(netcon.srcgid()))
                    connection_weights.append(netcon.weight[0])
                    if len(netcon.weight) == 3:
                        connection_weights_upd.append(netcon.weight[1])
                    else:
                        connection_weights_upd.append(np.nan)
            for external_id in cell_info.external_netcons.keys():
                external_cell_info = cell_info.external_netcons[external_id]
                for (idx,(presynaptic_gid, ncs, compartment, compartment_idx)) in enumerate(external_cell_info):
                    for netcon in ncs:
                        compartment_idxs.append(compartment_idx)
                        src_gids.append(int(netcon.srcgid()))
                        connection_weights.append(netcon.weight[0])
                        if len(netcon.weight) == 3: 
                            connection_weights_upd.append(netcon.weight[1])
                        else:
                            connection_weights_upd.append(np.nan)
            connection_info = np.core.records.fromarrays((np.asarray(connection_weights, dtype=np.float32),
                                                          np.asarray(connection_weights_upd, dtype=np.float32),
                                                          np.asarray(src_gids, dtype=np.uint32),
                                                          np.asarray(compartment_idxs, dtype=np.uint16)),
                                                         names='weights,weights_upd,src_gids,compartment_idxs')
            complete_weights[str(cell_gid)] = connection_info

    all_complete_weights = pc.py_gather(complete_weights, 0)

    if pc.id() == 0:
        complete_weights = {}
        for d in all_complete_weights:
            complete_weights.update(d)
        np.savez(save_filepath, **complete_weights)
        
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
