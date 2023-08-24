
import numpy as np
import yaml
import os
import sys
import pickle

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
        self.netstims = {}
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
                self.pc.spike_record(gid, self.external_spike_time_recs[cidx],
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
                
    def build_internal_netcons(self, internal_adj_matrices, seed=1e6):
        rnd = np.random.RandomState(seed=int(seed))
        src_population_ids = list(internal_adj_matrices.keys())
        for src_pop_id in src_population_ids:
            src_offset = self.ctype_offsets[src_pop_id]
            dst_population_ids = list(internal_adj_matrices[src_pop_id].keys())
            for dst_pop_id in dst_population_ids:
                adj_matrix = internal_adj_matrices[src_pop_id][dst_pop_id]
                dst_neurons = self.neurons[dst_pop_id] # PYR to PVBC, for example
                
                synapse_information = self.cell_params['internal connectivity'][src_pop_id][dst_pop_id]['synapse']
                for i in range(adj_matrix.shape[0]):
                    src_gid = i + src_offset
                    for j in range(adj_matrix.shape[1]):
                        nconnections = adj_matrix[i,j]
                        if nconnections == 0: continue

                        if not ((j in dst_neurons) and self.pc.gid_exists(dst_neurons[j].gid)):
                            continue
                        
                        compartments     = synapse_information['compartments']
                        rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                        chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]
                        
                        for con_num in range(nconnections):
                            compartment = chosen_compartments[con_num]
                            ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[j], 
                                                synapse_information, compartment, self.cell_params)

                            dst_neurons[j].internal_netcons.append( (src_gid, ncs, compartment) )

    def build_external_netcons(self, src_pop_id, external_adj_matrices, seed=1e7):
        seed = int(seed) + src_pop_id
        rnd  = np.random.RandomState(seed=seed)
        
        src_offset = self.ctype_offsets[src_pop_id]
        dst_population_ids = list(external_adj_matrices.keys())
        for dst_pop_id in dst_population_ids:
            adj_matrix = external_adj_matrices[dst_pop_id]
            
            dst_neurons = self.neurons[dst_pop_id] # PYR to PVBC, for example
            synapse_information = self.cell_params['external connectivity'][src_pop_id][dst_pop_id]['synapse']
            for i in range(adj_matrix.shape[0]):
                src_gid = i + src_offset
                for j in range(adj_matrix.shape[1]):
                    nconnections = adj_matrix[i,j]
                    if nconnections == 0: continue
                    compartments     = synapse_information['compartments']
                    rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                    chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]

                    if not ((j in dst_neurons) and self.pc.gid_exists(dst_neurons[j].gid)):
                        continue
                    
                    for con_num in range(nconnections):
                        compartment = chosen_compartments[con_num]
                        ncs = create_netcon(self.pc, src_pop_id, dst_pop_id, src_gid, dst_neurons[j], 
                                            synapse_information, compartment, self.cell_params)
                        if src_pop_id not in self.neurons[dst_pop_id][j].external_netcons: 
                            self.neurons[dst_pop_id][j].external_netcons[src_pop_id] = []
                        self.neurons[dst_pop_id][j].external_netcons[src_pop_id].append( (src_gid, ncs, compartment) )
                        
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
                        compartment = chosen_compartments[con_num]
                        ## TODO: provide source gids
                        ncs = create_netcon(self.pc, 'Septal', dst_pop_id,
                                            self.neurons['Septal'][i],
                                            self.neurons[dst_pop_id][j], 
                                            synapse_information, compartment, self.cell_params)
                        self.neurons[dst_pop_id][j].internal_netcons.append( (self.neurons['Septal'][i].gid, ncs, compartment) )

    def get_cell_spikes(self, group_id):
        neurons = self.neurons[group_id]
        spike_times = []
        for k in list(neurons.keys()):
            cell = neurons[k]
            spike_times.append(list(cell.spike_times))
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
                    
        
        

        
        
