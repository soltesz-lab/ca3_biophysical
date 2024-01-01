
' Author: Darian Hadjiabadi '

import numpy as np
from neuron import h

class CCKBC(object):

    def __init__(self, gid):
        self.gid = gid
        self.prelist   = []
        self.all        = None
        self.soma       = None
        self.radT2      = None
        self.radM2      = None
        self.radt2      = None
        self.lmM2       = None
        self.lmt2       = None
        self.radT1      = None
        self.radM1      = None
        self.radt1      = None
        self.lmM1       = None
        self.lmt1       = None
        self.oriT1      = None
        self.oriM1      = None
        self.orit1      = None
        self.oriT2      = None
        self.oriM2      = None
        self.orit2      = None
        
        self.synGroups = {}
        self.synGroups['AMPA'] = {}
        self.synGroups['GABAA'] = {}
        self.internal_netcons = []
        self.external_netcons = {}

        self.init()
        
        self.spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

        

    def init(self):
        self.topol()
        self.subsets()
        self.geom()
        self.biophys()
        #self.geom_nseg()
        self.synapses()



    def topol(self):
        self.soma       = h.Section(name='soma', cell=self)
        self.radT2      = h.Section(name='radT2', cell=self)
        self.radM2      = h.Section(name='radM2', cell=self)
        self.radt2      = h.Section(name='radt2', cell=self)
        self.lmM2       = h.Section(name='lmM2', cell=self)
        self.lmt2       = h.Section(name='lmt2', cell=self)
        self.radT1      = h.Section(name='radT1', cell=self)
        self.radM1      = h.Section(name='radM1', cell=self)
        self.radt1      = h.Section(name='radt1', cell=self)
        self.lmM1       = h.Section(name='lmM1', cell=self)
        self.lmt1       = h.Section(name='lmt1', cell=self)
        self.oriT1      = h.Section(name='oriT1', cell=self)
        self.oriM1      = h.Section(name='oriM1', cell=self)
        self.orit1      = h.Section(name='orit1', cell=self)
        self.oriT2      = h.Section(name='oriT2', cell=self)
        self.oriM2      = h.Section(name='oriM2', cell=self)
        self.orit2      = h.Section(name='orit2', cell=self)

    def subsets(self):
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def geom(self):
        self.soma.L = 20
        self.soma.diam = 10
    
    def biophys(self):      
        
        Vrest=-61
        celsius = 34.0  

        # Membrane resistance in ohm*cm2
        RmDend = 27000 #// 30000
        RmSoma = 27000 #// 27000 

        # Membrane capacitance in uF/cm2
        CmSoma= 1.0
        CmDend = 1.0

        # Axial resistance in ohm*cm
        RaDend= 150 #//75*3
        RaSoma= 150 #//75*3	

        # Calcium concentrations in mM
        ca_outside = 2
        ca_inside = 5.0e-6
        catau = 10

        # reversal potentials in mV
        ekval = -90	
        enaval = 55
        eHCNval = -40
        ecaval = 8.314*(273.15+celsius)/(2*9.649e4)*np.log(ca_outside/ca_inside)*1000 # about 170, otherwise set to 130

        if (Vrest<ekval): Vrest=ekval # Cell cannot rest lower than K+ reversal potential
        if (Vrest>enaval): Vrest=enaval # Cell cannot rest higher than Na+ reversal potential
        eleakval = Vrest
        
        
        gNav     = 0.005*1.8 #// soma: // 0.12 //original 0.030 to .055 ; lm: //0.5  	//original 0.015
        gKdr     = 0.003    #// Delayed rectifier potassium
        gKGroup  = 0.0011 #//0.1465/1
        gKvA     = 0.0010 #// Proximal A-type potassium
        gHCN     = 0.00002 #// HCN (hyperpolarization-activated cyclic nucleotide-gated channel)
        gCavN    = 0.00001 #//   N-type calcium
        gCavL    = 0.00025 #//  L-type calcium
        gKvCaB    = 0.0001 #// Big potassium channel: voltage and calcium gated 
        gKCaS    = 0.02 #//  Small potassium channel: calcium gated	
        gKvM     = 0.06
        for sec in self.all:
            sec.Ra = RaSoma
            sec.insert('ch_KvA')
            for seg in sec:
                seg.ch_KvA.gmax = gKvA
        
            sec.insert('ch_CavN')
            for seg in sec:
                seg.ch_CavN.gmax = gCavN
        
            sec.insert('ch_CavL')
            for seg in sec:
                seg.ch_CavL.gmax = gCavL
        
            sec.insert('ch_KCaS')
            for seg in sec:
                seg.ch_KCaS.gmax = gKCaS
            
            sec.insert('ch_KvCaB')
            for seg in sec:
                seg.ch_KvCaB.gmax = gKvCaB
            
            sec.insert('ch_HCN')
            for seg in sec:
                seg.ch_HCN.gmax = gHCN
               
        
        self.soma.insert('ch_Navcck')
        for seg in self.soma:
            seg.ch_Navcck.gmax = gNav
        self.soma.insert('ch_Kdrfast')
        for seg in self.soma:
            seg.ch_Kdrfast.gmax = gKdr
        self.soma.insert('ch_KvGroup')
        for seg in self.soma:
            seg.ch_KvGroup.gmax = gKGroup
        
        self.soma.insert('ch_KvM')
        for seg in self.soma:
            seg.ch_KvM.gbar = gKvM
        
        self.soma.insert('ch_leak')
        for seg in self.soma:
            seg.ch_leak.gmax = 1./RmSoma
        self.soma.cm  = CmSoma
        self.soma.ena = enaval
        
        for sec in self.all:
            sec.ek = ekval
            sec.eca = ecaval
            sec.e_ch_leak = eleakval
            #sec.cao_iconc_Ca = ca_outside
        
        

    def synapses(self):
            syn_ = h.MyExp2Syn(self.soma(0.5))
            syn_.tau1 = 2.0
            syn_.tau2 = 6.3
            syn_.e = 0
            self.synGroups['AMPA']['soma'] = [syn_]


            syn_ = h.MyExp2Syn(self.soma(0.5))
            syn_.tau1 = 0.287
            syn_.tau2 = 2.67
            syn_.e = -75
            self.synGroups['GABAA']['soma'] = [syn_]