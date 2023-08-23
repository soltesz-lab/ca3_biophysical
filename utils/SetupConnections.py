import numpy as np
import yaml
from copy import deepcopy
import random
import traceback
import matplotlib.pyplot as plt
from neuron import h


def place_field_fr(center, spatial_bins, max_fr, min_fr, diameter):
    c      = diameter / 4.3
    fnc = max_fr * np.exp(-( (spatial_bins-center) / (c)) ** 2.)
    return fnc



def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, delay=0.0, refractory=3., generator=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'np.random.RandomState()'
    :return: list of m spike times (ms)
    """
    tt = deepcopy(t)
#     fr_dt = tt[1]-tt[0]
#     tt += delay
#     expanded_t    = np.arange(0,delay,step=fr_dt)
#     expanded_rate = [1.0 for _ in range(len(expanded_t))]
#     rate = np.asarray(expanded_rate + list(rate), dtype='float32')
#     tt    = np.asarray(list(expanded_t) + list(tt), dtype='float32')
    min_fr = np.min(rate)
    if generator is None:
        generator = random
    interp_t = np.arange(tt[0], tt[-1] + dt, dt)
    try:
        interp_rate = np.interp(interp_t, tt, rate)
    except Exception:
        print('t shape: %s rate shape: %s' % (str(tt.shape), str(rate.shape)))
    
    delay_t = np.arange(-delay, 0, dt)
    delay_r = np.ones(len(delay_t))*2.
    
    interp_t = np.concatenate((delay_t, interp_t))
    interp_t += delay
    interp_rate = np.concatenate((delay_r, interp_rate)) 
    
    interp_rate /= 1000.
    spike_times = []
    non_zero = np.where(interp_rate > 1.e-100)[0]
    if len(non_zero) == 0:
        return spike_times
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    max_rate = np.max(interp_rate)
    if not max_rate > 0.:
        return spike_times
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.uniform(0.0, 1.0)
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI // dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.uniform(0.0, 1.0) <= interp_rate[i] / max_rate) and ISI_memory >= 0.:
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return np.asarray(spike_times, dtype='float32')



class Arena(object):
    
    def __init__(self, params_filepath, super_arena_flanks=(0,0), theta_modulation={}):
        self.pc = h.ParallelContext()
        self.params = {}
        self.params_filepath = params_filepath
        self._read_arena_params()
        self._read_arena_cell_params()
        self.arena_rnd = np.random.RandomState(seed=self.params['Arena']['random seed'])
        
        self.arena_size = self.params['Arena']['arena size']
        self.bin_size   = self.params['Arena']['bin size']
        self.arena_map  = np.arange(0, self.arena_size,step=self.bin_size)
        self.cell_information  = {}

    def _read_arena_params(self):
        self.params['Arena'] = {}
        with open(self.params_filepath, 'r') as f:
            fparams = yaml.load(f, Loader=yaml.FullLoader)           
            self.params['Arena'] = fparams['Arena']
    def _read_arena_cell_params(self):
        self.params['Spatial'] = {}
        with open(self.params_filepath, 'r') as f:
            fparams = yaml.load(f, Loader=yaml.FullLoader)
            self.params['Spatial'] = fparams['Arena Cells']
          
    def generate_population_firing_rates(self):
        
        for population_name in self.params['Spatial'].keys():
            self.cell_information[population_name] = {}
            current_population = self.params['Spatial'][population_name]

            self.cell_information[population_name]['id']     = current_population['id']
            self.cell_information[population_name]['ncells'] = current_population['ncells']
            somatic_positions = generate_soma_positions(current_population['ncells'])

            ncells = current_population['ncells']
            self.cell_information[population_name]['cell info'] = {}
            for gid in range(ncells):
                self.cell_information[population_name]['cell info'][gid] = {}
                self.cell_information[population_name]['cell info'][gid]['soma position'] = somatic_positions[gid]

            if 'place' in current_population:
                self.cell_information[population_name]['spatial type'] = 'place'
                field_centers = soma_positions_to_field_center(somatic_positions, self.arena_size)
                for gid in range(int(self.pc.id()), ncells, int(self.pc.nhost())):
                    self.cell_information[population_name]['cell info'][gid]['field center'] = field_centers[gid]


                peak_rate = current_population['place']['peak rate']
                min_rate  = current_population['place']['min rate']
                diameter  = current_population['place']['diameter']
                place_firing_rates = generate_place_firing_maps(field_centers, peak_rate, min_rate, diameter, self.arena_map)
                for gid in range(int(self.pc.id()), ncells, int(self.pc.nhost())):
                    self.cell_information[population_name]['cell info'][gid]['firing rate'] = place_firing_rates[gid]
                    
            elif 'grid' in current_population:
                self.cell_information[population_name]['spatial type'] = 'grid'
                field_centers = soma_positions_to_field_center(somatic_positions, self.arena_size)
                for gid in range(int(self.pc.id()), ncells, int(self.pc.nhost())):
                    self.cell_information[population_name]['cell info'][gid]['field center'] = field_centers[gid]

                peak_rate = current_population['grid']['peak rate']
                min_rate  = current_population['grid']['min rate']
                diameter  = current_population['grid']['diameter']
                gap       = current_population['grid']['gap']
                
                grid_firing_rates = generate_grid_firing_maps(field_centers, peak_rate, min_rate, diameter, gap, self.arena_map)
                for gid in range(int(self.pc.id()), ncells, int(self.pc.nhost())):
                    self.cell_information[population_name]['cell info'][gid]['firing rate'] = grid_firing_rates[gid]
            
            
    def generate_cue_firing_rates(self, population, percent_cue, seed=1e9):
            rnd = np.random.RandomState(seed=int(seed))
            noise_fr    = self.params['Spatial'][population]['noise']['min rate']
            ncells      = self.params['Spatial'][population]['ncells']
            
            ncue_cells = int(ncells*percent_cue)
            cells_cued = rnd.choice(np.arange(ncells), size=(ncue_cells,), replace=False)
            min_fr=self.params['Spatial'][population]['cue']['min rate']
            max_fr=self.params['Spatial'][population]['cue']['peak rate']
            diameter=self.params['Spatial'][population]['cue']['diameter']
            cue_firing_rates = []
            for i in range(ncells):
                cue_fr = None
                if i in cells_cued:
                    cue_fr = place_field_fr(int(self.arena_size/2), self.arena_map, max_fr, diameter)
                else:
                    cue_fr = [noise_fr for _ in range(len(self.arena_map))]
                cue_fr = np.asarray(cue_fr, dtype='float32')
                cue_fr[cue_fr<=min_fr] = min_fr
                cue_firing_rates.append(cue_fr)
            self.cell_information[population] = {}
            self.cell_information[population]['ncells'] = ncells
            self.cell_information[population]['id'] = self.params['Spatial'][population]['id']
            self.cell_information[population]['cell info'] = {}
            for (idx,cfr) in enumerate(cue_firing_rates):
                self.cell_information[population]['cell info'][idx] = {}
                self.cell_information[population]['cell info'][idx]['firing rate'] = cfr
                
                
    def generate_spike_times(self, population, dt=0.05, delay=0, cued=False):
         
        population_info = self.cell_information[population]
        ncells = population_info['ncells']
        gids   = range(int(self.pc.id()), ncells, int(self.pc.nhost()))
        
        nfr      = self.params['Spatial'][population]['noise']['min rate']
        noise_fr = [nfr for _ in range(len(self.arena_map))]
        firing_rates = {}
        for gid in gids:
            try:
                fr = population_info['cell info'][gid]['firing rate']
            except:
                fr = noise_fr
            firing_rates[gid] = fr
                
        #self.arena_size, self.bin_size 
        mouse_speed = self.params['Arena']['mouse speed']
        lap_information = self.params['Arena']['lap information']
        nlaps           = lap_information['nlaps']
        is_spatial      = lap_information['is spatial']
        
        start_time = 0
        end_time  = nlaps * self.arena_size / mouse_speed
        bin2times = np.arange(0, end_time, step=self.bin_size/mouse_speed, dtype='float32') * 1000.
    
        if cued:
            self.cued_positions  = np.linspace(12.5, self.arena_size-12.5, np.sum(is_spatial))
            self.random_cue_locs = np.arange(len(self.cued_positions))
            self.arena_rnd.shuffle(self.random_cue_locs)
            print(self.random_cue_locs)
        for (gid, fr) in firing_rates.items():
            current_full_fr = []
            online_number = 0
            for n in range(nlaps):
                if not is_spatial[n]:
                    if population == 'MF': current_full_fr.extend(np.multiply(noise_fr, 1.0))
                    else: current_full_fr.extend(noise_fr)
                else: 
                    if cued:
                        random_position = self.cued_positions[self.random_cue_locs[online_number]]
                        to_roll = int( ( self.arena_size/2 - random_position) / (self.arena_map[1]-self.arena_map[0]) )
                        current_full_fr.extend(np.roll(fr, to_roll))
                    else:
                        current_full_fr.extend(fr)
                    online_number += 1
            current_full_fr = np.asarray(current_full_fr, dtype='float32')
            if (bin2times.shape[0] > current_full_fr.shape[0]): bin2times = bin2times[:-1]
            spike_times = np.asarray(get_inhom_poisson_spike_times_by_thinning(current_full_fr, bin2times, dt=dt, delay=delay),
                                     dtype='float32')
            self.cell_information[population]['cell info'][gid]['spike times'] = spike_times

            
def generate_soma_positions(ncells, maxpos=1.):
    positions = np.linspace(0, maxpos, ncells)
    return { i: positions[i] for i in range(ncells) }

def soma_positions_to_field_center(volume_positions, arena_size):
    gids       = list(volume_positions.keys())
    volume_end = np.max(list(volume_positions.values()))
    scaling    = arena_size / float(volume_end)
    
    spatial_positions = {}
    for gid in gids:
        gid_pos     = volume_positions[gid]
        spatial_pos = gid_pos * scaling
        spatial_positions[gid] = spatial_pos
    return spatial_positions


def generate_place_firing_maps(field_centers, peak_rate, min_rate, diameter, arena_map):
    firing_rates = {}
    for gid in list(field_centers.keys()):
        current_center = field_centers[gid]
        fr = np.asarray(place_field_fr(current_center, arena_map, peak_rate, diameter), dtype='float32')
        fr[fr<=min_rate] = min_rate
        firing_rates[gid] = fr
    return firing_rates
   
    
def generate_grid_firing_maps(field_centers, peak_rate, min_rate, diameter, gap, arena_map):
    firing_rates = {}
    for gid in list(field_centers.keys()):
        current_center = field_centers[gid]
        current_firing_rate    = np.asarray(place_field_fr(current_center, arena_map, peak_rate, diameter), dtype='float32')
        arena_min, arena_max = np.min(arena_map), np.max(arena_map)
        current_pos = current_center - gap
        while (current_pos >= 0):
            hopped_fr = np.asarray(place_field_fr(current_pos, arena_map, peak_rate, diameter), dtype='float32')
            current_firing_rate += hopped_fr
            current_pos -= gap
        current_pos = current_center + gap
        while (current_pos <= arena_max):
            hopped_fr = np.asarray(place_field_fr(current_pos, arena_map, peak_rate, diameter), dtype='float32')
            current_firing_rate += hopped_fr
            current_pos += gap
        current_firing_rate[current_firing_rate <= min_rate] = min_rate
        firing_rates[gid] = current_firing_rate
    return firing_rates
                                  
        
        
        

def place_field_fr(center, spatial_bins, max_fr, diameter):
    c      = diameter / 4.3
    fnc = max_fr * np.exp(-( (spatial_bins-center) / (c)) ** 2.)
    return fnc  
    
    
    
#####

class WiringDiagram(object):
    
    def __init__(self, params_filepath, place_information):
        self.params = None
        self._read_params_filepath(params_filepath)
        self.internal_con_rnd = np.random.RandomState(seed=self.params['internal seed'])
        self.external_con_rnd = np.random.RandomState(seed=self.params['external seed'])
        self.septal_con_rnd   = np.random.RandomState(seed=self.params['septal seed'])
        
        place_ids   = place_information.get('place ids', [])
        place_fracs = place_information.get('place fracs', [])
        
        self.wiring_information = {}
        pops, ncells = self.params['cells'], self.params['ncells']
        self.pop2id = {pops[i]: i for i in range(len(pops))}
        self.pops   = pops
        self.place_information = {}
        
        for (i, pop) in enumerate(pops):
            self.wiring_information[pop] = {}
            self.wiring_information[pop]['ncells'] = ncells[i]
            self.wiring_information[pop]['cell info'] = {}
            
            for gid in np.arange(ncells[i]):
                self.wiring_information[pop]['cell info'][gid] = {}

            place_gids = []
            if i in place_ids:
                frac_place = place_fracs[place_ids.index(i)]
                print('frac place', frac_place)
                is_place = self.internal_con_rnd.choice([0,1],p=[1.0-frac_place, frac_place], size=(ncells[i],))
                for (gid,ip) in enumerate(is_place):
                    self.wiring_information[pop]['cell info'][gid]['place'] = ip
                    if ip: place_gids.append(gid)
            else: place_gids = np.arange(ncells[i])
           
            
            if ncells[i] < 10:
                e1 = 1./(2.*ncells[i]) 
                soma_coordinates = np.linspace(e1, 1.-e1, ncells[i])
            else:
                soma_coordinates = np.linspace(0, 1, len(place_gids))
            
            for (didx,gid) in enumerate(place_gids):
                self.wiring_information[pop]['cell info'][gid]['soma position'] = soma_coordinates[didx]
            
            notplace_gids = list( set(np.arange(ncells[i])) - set(place_gids) )
            
            if i in place_ids:
                self.place_information[i] = {}
                self.place_information[i]['place']     = place_gids
                self.place_information[i]['not place'] = notplace_gids
            
            
            #left_behind_coordinates = self.internal_con_rnd.uniform(0., 1., size=(len(notplace_gids),))
            left_behind_coordinates = np.linspace(0.1, 0.9, num=len(notplace_gids))
            for (didx,gid) in enumerate(notplace_gids):
                self.wiring_information[pop]['cell info'][gid]['soma position'] = left_behind_coordinates[didx]
                
                
 
                
        
    def generate_internal_connectivity(self):
        self.internal_adj_matrices = {}
        for popA in self.pops:
            popA_id = self.pop2id[popA]
            self.internal_adj_matrices[popA_id] = {}
            for popB in self.pops:   
                popB_id = self.pop2id[popB]
                popA_pos = self._get_soma_coordinates(self.wiring_information[popA]['cell info'])
                popB_pos = self._get_soma_coordinates(self.wiring_information[popB]['cell info'])
                nA, nB = self.wiring_information[popA]['ncells'], self.wiring_information[popB]['ncells']
                try:
                    convergence = self.params['internal connectivity'][popA_id][popB_id]['probability']
                    same_pop = False
                    if popA == popB: same_pop=True
                    if popA_id == 0 and popB_id == 0:
                        alpha = 0.0125
                        am = self.create_adjacency_matrix(popA_pos, popB_pos, nA, nB, convergence, self.internal_con_rnd, 
                                                          inv_func=('exp', alpha), same_pop=same_pop, src_id=popA_id)
                    else:
                        am = self.create_adjacency_matrix(None, None, nA, nB, convergence, self.internal_con_rnd, same_pop=same_pop)   
                    self.internal_adj_matrices[popA_id][popB_id] = am
                except:
                    continue
                   #print('no connection between src %s and dst %s' % (popA, popB) )

                    
    def generate_external_connectivity(self, external_information, **kwargs):
        
        
        place_information  = kwargs['place information']
        external_place_ids = kwargs['external place ids']
        
        cue_information = kwargs['cue information']
        external_cue_ids = kwargs['external cue ids']
        
        external_pops = list(external_information.keys())
        external_ids  = [external_information[pop]['id'] for pop in external_pops]
        self.external_pop2id = {pop:i for (pop,i) in list(zip(external_pops, external_ids))}
        
        self.external_adj_matrices = {}
        for src_id,src_pop in list(zip(external_ids,external_pops)):
            self.external_adj_matrices[src_id] = {}
            for dst_pop in self.pops:
                dst_pop_id = self.pop2id[dst_pop]
                try:
                    src_pos = self._get_soma_coordinates(external_information[src_pop]['cell info'])
                except:
                    src_pos = None
                dst_pos = self._get_soma_coordinates(self.wiring_information[dst_pop]['cell info'])
                nsrc, ndst = external_information[src_pop]['ncells'], self.wiring_information[dst_pop]['ncells']
                
                if dst_pop_id not in self.params['external connectivity'][src_id]: continue
                convergence = self.params['external connectivity'][src_id][dst_pop_id]['probability']

                place_connection_flag = dst_pop_id in place_information and src_id in external_place_ids
                cue_connection_flag = dst_pop_id in cue_information and src_id in external_cue_ids

                dst_gids_to_connect_to = []
                
                if place_connection_flag:
                    dst_gids_to_connect_to.append(place_information[dst_pop_id]['place'])
                    
                if cue_connection_flag:
                    dst_gids_to_connect_to.append(cue_information[dst_pop_id]['not place'])

                if not (place_connection_flag or cue_connection_flag):
                    dst_gids_to_connect_to.append(np.arange(ndst))

                dst_gids_to_connect_to = np.concatenate(dst_gids_to_connect_to)

                print(f"src_id = {src_id} dst_pop_id = {dst_pop_id} "
                      f"external_place_ids = {external_place_ids} external_cue_ids = {external_cue_ids} "
                      f"src_id in external_cue_ids = {src_id in external_cue_ids} "
                      f"src_id in external_place_ids = {src_id in external_place_ids} ")
                
                if dst_pop_id == 0 and src_id < 102: # For MF and MEC connections 0.0015
                    am = self.create_adjacency_matrix(src_pos, dst_pos, nsrc, ndst, convergence, self.external_con_rnd, 
                                                      inv_func=('exp', 0.00075), valid_gids=dst_gids_to_connect_to, src_id=src_id)
                else:
                    am = self.create_adjacency_matrix(None, None, nsrc, ndst, convergence, 
                                                      self.external_con_rnd, valid_gids=dst_gids_to_connect_to)  
                self.external_adj_matrices[src_id][dst_pop_id] = am

                    
    def generate_septal_connectivity(self):
        id2pop = {v: k for (k,v) in self.pop2id.items()}
        self.septal_adj_matrices = {}
        dst_ids = list(self.params['Septal']['connectivity'].keys())
        for dst_id in dst_ids:
            if dst_id not in list(id2pop.keys()): continue
            
            nsrc, ndst = self.params['Septal']['ncells'], self.wiring_information[id2pop[dst_id]]['ncells']
            try:
                convergence = self.params['Septal']['connectivity'][dst_id]['probability']
                am = self.create_adjacency_matrix(None, None, nsrc, ndst, convergence, self.septal_con_rnd, same_pop=False)
                self.septal_adj_matrices[dst_id] = am
            except:
                print('Septal error...', dst_id, convergence)
            
            
                
    def _get_soma_coordinates(self, gid_dict):
        pos = []
        for gid in np.sort(list(gid_dict.keys())):
            pos.append(gid_dict[gid]['soma position'])
        return pos
        
    def _read_params_filepath(self, params_filepath):
        with open(params_filepath, 'r') as f:
            fparams = yaml.load(f, Loader=yaml.FullLoader)           
            self.params = fparams['Circuit']

    def create_adjacency_matrix(self, src_coordinates, dst_coordinates, nsrc, ndst, convergence, rnd, 
                                inv_func=None, same_pop=False, valid_gids=None, src_id=None):
        
        if valid_gids is None: valid_gids = np.arange(ndst)
        adj_mat = np.zeros((nsrc, ndst), dtype='uint16')
        for d in range(ndst):
            if d not in valid_gids: continue
            if src_coordinates is not None and dst_coordinates is not None:

                dst_coord = dst_coordinates[d]
                distances = np.asarray([(dst_coord - src_coord)**2 for src_coord in src_coordinates])
                if inv_func[0] == 'inv':
                    inv_dist  = 1./(distances+1.0e-5)**inv_func[1] #2.0
                    #inv_dist = np.exp(-distances/0.01)
                elif inv_func[0] == 'exp':
                    inv_dist = np.exp(-distances/inv_func[1]) #0.01
                if same_pop: 
                    inv_dist[d] = 0.0
                
                effective_convergence = deepcopy(convergence) * self.params['scale']
                pcon      = inv_dist/(np.sum(inv_dist) + 1.0e-10)
                           
                if (src_id == 0) and (dst_coord < 0.1 or dst_coord > 0.90) and (d in self.place_information[0]['place']):
                    effective_convergence = int(effective_convergence*0.75)
                elif (src_id == 100 or src_id == 101) and (dst_coord < 0.1 or dst_coord > 0.90):
                    effective_convergence = int(effective_convergence*0.75)
                presynaptic_gids = rnd.choice(np.arange(nsrc), p=pcon, replace=False, size=(int(effective_convergence),))
#                 if (src_id == 0) and (d in self.place_information[0]['place']):
#                     if d > 0 and d - 1 not in presynaptic_gids:
#                         presynaptic_gids[0] = d - 1
#                     if d < ndst - 1 and d + 1 not in presynaptic_gids:
#                         presynaptic_gids[-1] = d + 1                  
                
            else:
                effective_convergence = deepcopy(convergence) * self.params['scale'] 
                presynaptic_gids = rnd.choice(np.arange(nsrc), replace=True, size=(effective_convergence,))
                
            for gid in presynaptic_gids:
                adj_mat[gid,d] += 1
        return adj_mat
                
    
    
