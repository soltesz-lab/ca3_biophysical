from neuron import h
import numpy as np
import pickle
        
class ca3pyrcell_v2(object):
    def __init__(self, gid, ctype, props_filename, syn_params):
        self.gid   = gid
        self.ctype = ctype
        self.syn_params = syn_params
        self.spike_threshold = -42.0
        
        self.soma = None
        self.axon = None
        self.oriensProximal = None
        self.oriensDistal   = None
        self.lucidum =  None
        self.radiatum = None
        self.lacunosumMEC = None
        self.lacunosumLEC = None
                
        self.generate_morphology()
        self.generate_biophysics(ctype, props_filename)
        
        self.synGroups = None
        self.generate_synapses()
        
        self.internal_netcons = []
        self.external_netcons = {}
        self.spike_detector = h.NetCon(self.axon(0.5)._ref_v, None, sec=self.axon)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)
    
    
    def generate_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        
        self.axon = h.Section(name='axon', cell=self)
        self.axon.connect(self.soma(1.0))
        #self.axon.connect(self.soma,1,0)
        
        self.oriensProximal = h.Section(name='oriensProximal', cell=self)
        #self.oriensProximal.connect(self.soma(1.0))
        #self.oriensProximal.connect(self.soma,1,0)
        
        self.oriensDistal = h.Section(name='oriensDistal', cell=self)
        #self.oriensDistal.connect(self.oriensProximal(1.0))
        #self.oriensDistal.connect(self.oriensProximal,1,0)
        
        self.lucidum = h.Section(name='lucidum', cell=self)
        #self.lucidum.connect(self.soma(0.0))
        #self.lucidum.connect(self.soma,1,0)
        
        self.radiatum = h.Section(name='radiatum', cell=self)
        #self.radiatum.connect(self.lucidum(1.0))
        #self.radiatum.connect(self.lucidum,1,0)

        self.lacunosumMEC = h.Section(name='lacunosumMEC', cell=self)
        #self.lacunosumMEC.connect(self.radiatum(1.0))
        #self.lacunosumMEC.connect(self.radiatum,1,0)
        
        self.lacunosumLEC = h.Section(name='lacunosumLEC', cell=self)
        #self.lacunosumLEC.connect(self.lacunosumMEC(1.0))
        #self.lacunosumLEC.connect(self.lacunosumMEC,1,0)
        
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)
        
    
    
    def generate_biophysics(self, ctype, props_filename):
        
        ghd   = 0.00001
        gna   = 0.022
        KMULT = 0.02
        gc    = 1.0e-5
        gahp  = 0.001
        epas  = -65.0
        gcal = gc
        gcan = gc
        gcat = gc
        RA   = 140.0
        Cm   = 0.72
        PPSpineAdj =1.0
        RCSpineAdj =2.0
        if ctype == 'B':
            gKc  = 5.0e-5
            gkdr = 0.005 #0.005
            gkm  = 0.017 * 1.7
            gkd  = 0.0 # 0.0015 for Ra 150 # 0.0028 for Ra 75 #0.0005 #0.0
        elif ctype == 'C':
            gKc  = 2.0e-5
            gkdr = 0.01
            gkm  = 0.017*1.25
            gkd  = 0.0
        elif ctype == 'D':
            gKc  = 5.0e-5
            gkdr = 0.01 #0.005
            gkm  = 0.015 #0.017
            gkd  = 0.1 # 0.0015 for Ra 150 # 0.0028 for Ra 75 #0.0005 #0.0
        if ctype == 'B':
            Rm = 71956.9
        elif ctype == 'C':
            Rm = 74719.20
        elif ctype == 'D':
            Rm = 68916.1
        
        self.soma.nseg = 1
        self.soma.L    = 11.22
        self.soma.diam = 13.21
        self.soma.cm   = 0.72
        
        self.soma.insert('ds_CA3')
        self.soma.insert('hd_CA3')
        self.soma.insert('na3_CA3')
        self.soma.insert('kdr_CA3')
        self.soma.insert('kap_CA3')
        self.soma.insert('km_CA3')
        if ctype == 'B' or ctype == 'C':
            self.soma.insert('kd_CA3')
        elif ctype == 'D':
            self.soma.insert('KdBG')
        self.soma.insert('cacum_CA3')
        for seg in self.soma:
            seg.cacum_CA3.depth=self.soma.diam/2
        self.soma.insert('cal_CA3')
        self.soma.insert('can_CA3')
        self.soma.insert('cat_CA3')
        self.soma.insert('cagk_CA3')
        #self.soma.insert('KahpM95_CA3')
        self.soma.ghdbar_hd_CA3=1.0e-5
        self.soma.gbar_na3_CA3=0.44
        self.soma.gkdrbar_kdr_CA3=0.077
        self.soma.gkabar_kap_CA3 = 0.047
        self.soma.gbar_km_CA3= 0.026
        if ctype == 'B' or ctype == 'C':
            self.soma.gkdbar_kd_CA3 = 0.0
        else:
            self.soma.gbar_KdBG = gkd
        self.soma.ehd_hd_CA3=-30.0
        self.soma.gcalbar_cal_CA3=1.1e-4
        self.soma.gcanbar_can_CA3=1.1e-4
        self.soma.gcatbar_cat_CA3=1.1e-4
        self.soma.gbar_cagk_CA3= 3.0e-5
        #self.soma.gbar_KahpM95_CA3 = gahp
        self.soma.insert('pas')
        self.soma.e_pas = -65.0
        self.soma.g_pas = 1.4e-5
        self.soma.Ra = 140.0
        self.soma.ek = -90.0
        self.soma.ena = 55.0
        self.soma.sh_na3_CA3 = 24
        self.soma.sh_kdr_CA3 = 24
        self.soma.sh_kap_CA3 = 24
        

        self.axon.nseg = 1
        self.axon.L = 97.09
        self.axon.diam = 1.03

        self.axon.insert('na3_CA3')
        self.axon.insert('kdr_CA3')
        self.axon.insert('kap_CA3')
        self.axon.insert('pas')
        self.axon.gbar_na3_CA3=0.11
        self.axon.gkdrbar_kdr_CA3=0.005
        self.axon.gkabar_kap_CA3 = 0.047
        self.axon.sh_kap_CA3=0
        self.axon.e_pas = -65.0
        self.axon.g_pas = 1.4e-5
        self.axon.Ra = 50.
        self.axon.cm = 0.72
        self.axon.ek = -90.0
        self.axon.ena = 55.0
        self.axon.sh_na3_CA3 = 24
        self.axon.sh_kdr_CA3 = 24
        self.axon.sh_kap_CA3 = 0
        
        
        with open(props_filename, "rb") as f:
            props = pickle.load(f, encoding='latin1')
        
        self.oriensProximal.nseg = 1#props['oriensProximal']['nseg']
        self.oriensProximal.L = 189.25 #props['oriensProximal']['L']
        self.oriensProximal.diam = 2.55 #props['oriensProximal']['d']
        self.oriensProximal.Ra = 135.69 #props['oriensProximal']['Ra']

        self.oriensProximal.insert('ds_CA3')
        self.oriensProximal.insert('hd_CA3')
        self.oriensProximal.insert('na3_CA3')
        self.oriensProximal.insert('kdr_CA3')
        self.oriensProximal.insert('kap_CA3')
        self.oriensProximal.insert('cacum_CA3')
        self.oriensProximal.insert('cal_CA3')
        self.oriensProximal.insert('can_CA3')
        self.oriensProximal.insert('cat_CA3')
        self.oriensProximal.insert('cagk_CA3')
        #self.oriensProximal.insert('KahpM95_CA3')
        self.oriensProximal.insert('pas')

        self.oriensProximal.ghdbar_hd_CA3 = 7.1e-5 #ghd*props['oriensProximal']['fact']
        self.oriensProximal.gbar_na3_CA3 = 0.16 #gna*props['oriensProximal']['fact']
        self.oriensProximal.gkdrbar_kdr_CA3 = 0.035 #gkdr*props['oriensProximal']['fact']
        self.oriensProximal.gkabar_kap_CA3 = 0.14 #KMULT*props['oriensProximal']['fact']
        self.oriensProximal.gcalbar_cal_CA3 = 7.1e-5 #gc*props['oriensProximal']['fact']
        self.oriensProximal.gcanbar_can_CA3 = 7.1e-5 #gc*props['oriensProximal']['fact']
        self.oriensProximal.gcatbar_cat_CA3 = 7.1e-5 #gc*props'oriensProximal']['fact']
        self.oriensProximal.gbar_cagk_CA3 = 3.5e-4 #gKc*props['oriensProximal']['fact']
        #self.oriensProximal.gbar_KahpM95_CA3 = gahp*props['oriensProximal']['fact']
        self.oriensProximal.cm = 10.17 #Cm*RCSpineAdj*props['oriensProximal']['fact']
       
        for seg in self.oriensProximal:
            seg.cacum_CA3.depth = self.oriensProximal.diam/2.
        self.oriensProximal.ek = -90.0
        self.oriensProximal.ena = 55.0
        self.oriensProximal.ehd_hd_CA3=-30.0
        self.oriensProximal.e_pas = epas
        self.oriensProximal.g_pas = 2.0e-4
        self.oriensProximal.sh_na3_CA3 = 24
        self.oriensProximal.sh_kdr_CA3 = 24
        self.oriensProximal.sh_kap_CA3 = 24
        

        self.oriensDistal.nseg = 1 #props['oriensDistal']['nseg']
        self.oriensDistal.L = 71.68 #props['oriensDistal']['L']
        self.oriensDistal.diam = 1.77 #props['oriensDistal']['d']
        self.oriensDistal.Ra = 135.66 #props['oriensDistal']['Ra']

        self.oriensDistal.insert('ds_CA3')
        self.oriensDistal.insert('hd_CA3')
        self.oriensDistal.insert('na3_CA3')
        self.oriensDistal.insert('kdr_CA3')
        self.oriensDistal.insert('kap_CA3')
        self.oriensDistal.insert('cacum_CA3')
        self.oriensDistal.insert('cal_CA3')
        self.oriensDistal.insert('can_CA3')
        self.oriensDistal.insert('cat_CA3')
        self.oriensDistal.insert('cagk_CA3')
        #self.oriensDistal.insert('KahpM95_CA3')
        self.oriensDistal.insert('pas')

        self.oriensDistal.ghdbar_hd_CA3 = 6.0e-5 #ghd*props['oriensDistal']['fact']
        self.oriensDistal.gbar_na3_CA3 = 0.13 #gna*props['oriensDistal']['fact']
        self.oriensDistal.gkdrbar_kdr_CA3 = 0.030 #gkdr*props['oriensDistal']['fact']
        self.oriensDistal.gkabar_kap_CA3 = 0.12 #KMULT*props['oriensDistal']['fact']
        self.oriensDistal.gcalbar_cal_CA3 = 6.0e-5 #gc*props['oriensDistal']['fact']
        self.oriensDistal.gcanbar_can_CA3 = 6.0e-5 #gc*props['oriensDistal']['fact']
        self.oriensDistal.gcatbar_cat_CA3 = 6.0e-5 #gc*props['oriensDistal']['fact']
        self.oriensDistal.gbar_cagk_CA3 = 3.0e-4 #gKc*props['oriensDistal']['fact']
        #self.oriensDistal.gbar_KahpM95_CA3 = gahp*props['oriensDistal']['fact']
        self.oriensDistal.g_pas = 1.7e-4 #1./Rm*RCSpineAdj*props['oriensDistal']['fact']
        self.oriensDistal.cm = 8.70 #Cm*RCSpineAdj*props['oriensDistal']['fact']

        for seg in self.oriensDistal:
            seg.cacum_CA3.depth = self.oriensDistal.diam / 2.
            
        self.oriensDistal.ek = -90.0
        self.oriensDistal.ena = 55.0
        self.oriensDistal.ehd_hd_CA3=-30.0
        self.oriensDistal.e_pas = epas
        self.oriensDistal.sh_na3_CA3 = 24
        self.oriensDistal.sh_kdr_CA3 = 24
        self.oriensDistal.sh_kap_CA3 = 24

        self.lucidum.nseg = 1 #props['lucidum']['nseg']
        self.lucidum.L = 92.27 #props['lucidum']['L']
        self.lucidum.diam = 6.41 #props['lucidum']['d']
        self.lucidum.Ra = 129.28 #props['lucidum']['Ra']

        self.lucidum.insert('hd_CA3')
        self.lucidum.insert('na3_CA3')
        self.lucidum.insert('kdr_CA3')
        self.lucidum.insert('kap_CA3')
        self.lucidum.insert('pas')
        self.lucidum.ehd_hd_CA3=-30.0
        self.lucidum.e_pas = epas
        self.lucidum.ek = -90.0
        self.lucidum.ena = 55.0
        self.lucidum.sh_na3_CA3 = 24
        self.lucidum.sh_kdr_CA3 = 24
        self.lucidum.sh_kap_CA3 = 24

        self.lucidum.gbar_na3_CA3 = 0.028 #gna*props['lucidum']['fact']
        self.lucidum.gkdrbar_kdr_CA3 = 0.0064 #gkdr*props['lucidum']['fact']
        self.lucidum.gkabar_kap_CA3 = 0.026 #KMULT*props['lucidum']['fact']
        self.lucidum.ghdbar_hd_CA3 = 1.28e-5 #ghd*props['lucidum']['fact']
        self.lucidum.g_pas = 3.6e-5 #1./Rm*RCSpineAdj*props['lucidum']['fact']
        self.lucidum.cm = 1.84 #Cm*RCSpineAdj*props['lucidum']['fact']
        
        self.radiatum.nseg = props['radiatum']['nseg']
        self.radiatum.L = props['radiatum']['L']
        self.radiatum.diam = props['radiatum']['d']
        self.radiatum.Ra = props['radiatum']['Ra']

        self.radiatum.insert('hd_CA3')
        self.radiatum.insert('na3_CA3')
        self.radiatum.insert('kdr_CA3')
        self.radiatum.insert('kap_CA3')
        self.radiatum.insert('pas')
        self.radiatum.ehd_hd_CA3=-30.0
        self.radiatum.e_pas = epas
        self.radiatum.ek = -90.0
        self.radiatum.ena = 55.0
        self.radiatum.sh_na3_CA3 = 24
        self.radiatum.sh_kdr_CA3 = 24
        self.radiatum.sh_kap_CA3 = 24

        self.radiatum.gbar_na3_CA3 = gna*props['radiatum']['fact']
        self.radiatum.gkdrbar_kdr_CA3 = gkdr*props['radiatum']['fact']
        self.radiatum.gkabar_kap_CA3 = KMULT*props['radiatum']['fact']
        self.radiatum.ghdbar_hd_CA3 = ghd*props['radiatum']['fact']
        self.radiatum.g_pas = 1./Rm*RCSpineAdj*props['radiatum']['fact']
        self.radiatum.cm = Cm*RCSpineAdj*props['radiatum']['fact']


        self.lacunosumMEC.nseg = props['lacunosumMEC']['nseg']
        self.lacunosumMEC.L = props['lacunosumMEC']['L']
        self.lacunosumMEC.diam = props['lacunosumMEC']['d']
        self.lacunosumMEC.Ra = props['lacunosumMEC']['Ra']

        self.lacunosumMEC.insert('hd_CA3')
        self.lacunosumMEC.insert('na3_CA3')
        self.lacunosumMEC.insert('kdr_CA3')
        self.lacunosumMEC.insert('kap_CA3')
        self.lacunosumMEC.insert('pas')
        self.lacunosumMEC.ehd_hd_CA3=-30.0
        self.lacunosumMEC.e_pas = epas
        self.lacunosumMEC.ek = -90.0
        self.lacunosumMEC.ena = 55.0
        self.lacunosumMEC.sh_na3_CA3 = 24
        self.lacunosumMEC.sh_kdr_CA3 = 24
        self.lacunosumMEC.sh_kap_CA3 = 24

        self.lacunosumMEC.gbar_na3_CA3 = gna*props['lacunosumMEC']['fact']
        self.lacunosumMEC.gkdrbar_kdr_CA3 = gkdr*props['lacunosumMEC']['fact']
        self.lacunosumMEC.gkabar_kap_CA3 = KMULT*props['lacunosumMEC']['fact']
        self.lacunosumMEC.ghdbar_hd_CA3 = ghd*props['lacunosumMEC']['fact']
        self.lacunosumMEC.g_pas = 1./Rm*PPSpineAdj*props['lacunosumMEC']['fact']
        self.lacunosumMEC.cm = Cm*PPSpineAdj*props['lacunosumMEC']['fact']

        self.lacunosumLEC.nseg = props['lacunosumLEC']['nseg']
        self.lacunosumLEC.L = props['lacunosumLEC']['L']
        self.lacunosumLEC.diam = props['lacunosumLEC']['d']
        self.lacunosumLEC.Ra = props['lacunosumLEC']['Ra']

        self.lacunosumLEC.insert('hd_CA3')
        self.lacunosumLEC.insert('na3_CA3')
        self.lacunosumLEC.insert('kdr_CA3')
        self.lacunosumLEC.insert('kap_CA3')
        self.lacunosumLEC.insert('pas')
        self.lacunosumLEC.ehd_hd_CA3=-30.0
        self.lacunosumLEC.e_pas = epas
        self.lacunosumLEC.ek = -90.0
        self.lacunosumLEC.ena = 55.0
        self.lacunosumLEC.sh_na3_CA3 = 24
        self.lacunosumLEC.sh_kdr_CA3 = 24
        self.lacunosumLEC.sh_kap_CA3 = 24

        self.lacunosumLEC.gbar_na3_CA3 = gna*props['lacunosumLEC']['fact']
        self.lacunosumLEC.gkdrbar_kdr_CA3 = gkdr*props['lacunosumLEC']['fact']
        self.lacunosumLEC.gkabar_kap_CA3 = KMULT*props['lacunosumLEC']['fact']
        self.lacunosumLEC.ghdbar_hd_CA3 = ghd*props['lacunosumLEC']['fact']
        self.lacunosumLEC.g_pas = 1./Rm*PPSpineAdj*props['lacunosumLEC']['fact']
        self.lacunosumLEC.cm = Cm*PPSpineAdj*props['lacunosumLEC']['fact']
        
        
    def generate_synapses(self):
        region_dict = {'axon': self.axon, 'soma': self.soma, 'oriensProximal': self.oriensProximal, 'oriensDistal': self.oriensDistal,
                       'lucidum': self.lucidum, 'radiatum': self.radiatum, 'lacunosumMEC': self.lacunosumMEC, 
                       'lacunosumLEC': self.lacunosumLEC}
        self._make_syn_groups()
        for syn_type in self.synGroups:
            for roi in region_dict:
                syn = self._create_synapse(syn_type, region_dict[roi], 0.5)
                if syn is not None:
                    self.synGroups[syn_type][roi].append(syn)
            
    def _make_syn_groups(self):
        region_list = ['axon', 'soma', 'oriensProximal', 'oriensDistal', 'lucidum', 'radiatum', 'lacunosumMEC', 'lacunosumLEC']
        self.synGroups = {} 
        self.synGroups['GABAA'] = {}
        for roi in region_list:
            self.synGroups['GABAA'][roi] = {}
            
        self.synGroups['NMDA'] = {}
        for roi in region_list:
            self.synGroups['NMDA'][roi] = []
            
        self.synGroups['STDPE2'] = {}
        for roi in region_list:
            self.synGroups['STDPE2'][roi] = []
        
    def _create_synapse(self, syn_type, sec_choice, seg_choice):       
        syn = None
        # populate excitatory synapses only at this stage
        if syn_type == 'STDPE2':
            syn = h.STDPE2(sec_choice(seg_choice))#h.AMPANMDA(sec_choice(seg_choice))
            syn.e = 0.0
            if sec_choice == 'oriensDistal':
                syn.tau1 = 0.5
                syn.tau2 = 8.77
            elif sec_choice == 'oriensProximal':
                syn.tau1 = 0.5
                syn.tau2 = 144.03
            if sec_choice == 'lucidum':
                syn.tau1 = 0.5
                syn.tau2 = 144.03    
            elif sec_choice == 'radiatum':
                syn.tau1 = 0.5
                syn.tau2 = 9.77
            elif sec_choice == 'lacunosumMEC':
                syn.tau1 = 0.5
                syn.tau2 = 12.66
            elif sec_choice == 'lacunosumLEC':
                syn.tau1 = 0.5
                syn.tau2 = 12.66  
                
            syn.p   = self.syn_params.get('potentiation', 1.0)
            syn.d   = self.syn_params.get('depression', 0.4)
            syn.thresh = self.syn_params.get('thresh', self.spike_threshold) #-55?

        elif syn_type == 'NMDA':
            syn = h.NMDA(sec_choice(seg_choice))
            syn.tcon = 2.3
            syn.tcoff = 100.0
            syn.gNMDAmax = 1.0
        return syn
