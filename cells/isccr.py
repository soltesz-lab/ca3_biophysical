
' Author: Darian Hadjiabadi '

import numpy as np
from neuron import h, gui

class ISCCR(object):

    def __init__(self, gid):
        self.gid = gid
        self.prelist   = []
        self.all        = None
        self.soma       = None
        self.radProx1   = None
        self.radMed1    = None
        self.radDist1   = None
        self.lmM1       = None
        self.lmt1       = None
        self.radProx2   = None
        self.radMed2    = None
        self.radDist2   = None
        self.lmM2       = None
        self.lmt2       = None
        self.oriProx1   = None
        self.oriMed1    = None
        self.oriDist1   = None
        self.oriProx2   = None
        self.oriMed2    = None
        self.oriDist2   = None
 
        self.internal_netcons = []
        self.external_netcons = {}
        self.synGroups = {}
        self.synGroups['AMPA'] = {}
        self.synGroups['GABAA'] = {}
        self.synGroups['GABAB'] = {}

        self.init()
        
        self.spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)
        self.spike_threshold = -45

    def init(self):
        self.topol()
        self.subsets()
        self.geom()
        self.biophys()
        self.geom_nseg()
        self.synapses()

    def topol(self):
        self.soma       = h.Section(name='soma', cell=self)
        self.radProx1   = h.Section(name='radProx1', cell=self)
        self.radMed1    = h.Section(name='radMed1', cell=self)
        self.radDist1   = h.Section(name='radDist1', cell=self)
        self.lmM1       = h.Section(name='lmM1', cell=self)
        self.lmt1       = h.Section(name='lmt1', cell=self)
        self.radProx2   = h.Section(name='radProx2', cell=self)
        self.radMed2    = h.Section(name='radMed2', cell=self)
        self.radDist2   = h.Section(name='radDist2', cell=self)
        self.lmM2       = h.Section(name='lmM2', cell=self)
        self.lmt2       = h.Section(name='lmt2', cell=self)
        self.oriProx1   = h.Section(name='oriProx1', cell=self)
        self.oriMed1    = h.Section(name='oriMed1', cell=self)
        self.oriDist1   = h.Section(name='oriDist1', cell=self)
        self.oriProx2   = h.Section(name='oriProx2', cell=self)
        self.oriMed2    = h.Section(name='oriMed2', cell=self)
        self.oriDist2   = h.Section(name='oriDist2', cell=self)
        
        self.radProx1.connect(self.soma(0))
        self.radMed1.connect(self.radProx1(1))
        self.radDist1.connect(self.radMed1(1))
        self.lmM1.connect(self.radDist1(1))
        self.lmt1.connect(self.lmM1(1))
        self.radProx2.connect(self.soma(1))
        self.radMed2.connect(self.radProx2(1))
        self.radDist2.connect(self.radMed2(1))
        self.lmM2.connect(self.radDist2(1))
        self.lmt2.connect(self.lmM2(1))
        self.oriProx1.connect(self.soma(0))
        self.oriMed1.connect(self.oriProx1(1))
        self.oriDist1.connect(self.oriMed1(1))
        self.oriProx2.connect(self.soma(1))
        self.oriMed2.connect(self.oriProx2(1))
        self.oriDist2.connect(self.oriMed2(1))
        
        self.basic_shape()

    def basic_shape(self):
        pass

    def subsets(self):
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def geom(self):
        self.soma.L = 20
        self.soma.diam = 10.

        self.radProx1.L = 50.
        self.radProx1.diam = 4.
        
        self.radMed1.L = 50.
        self.radMed1.diam = 3.
        
        self.radDist1.L = 100.
        self.radDist1.diam = 2.
        
        self.lmM1.L = 50.
        self.lmM1.diam = 1.5
        
        self.lmt1.L = 50.
        self.lmt1.diam = 1.
        
        self.radProx2.L = 50.
        self.radProx2.diam = 4.
        
        self.radMed2.L = 50.
        self.radMed2.diam = 3.
        
        self.radDist2.L = 100.
        self.radDist2.diam = 2.
        
        self.lmM2.L = 50.
        self.lmM2.diam = 1.5
        
        self.lmt2.L = 50.
        self.lmt2.diam = 1.
        
        self.oriProx1.L = 50.
        self.oriProx1.diam = 2.
        
        self.oriMed1.L = 50.
        self.oriMed1.diam = 1.5
        
        self.oriDist1.L = 50.
        self.oriDist1.diam = 1.
        
        self.oriProx2.L= 50.
        self.oriProx2.diam = 2.
        
        self.oriMed2.L = 50.
        self.oriMed2.diam = 1.5
        
        self.oriDist2.L = 50.
        self.oriDist2.diam = 1.

    def biophys(self):

        soma_nafcr   = 0.015
        soma_kdrcr   = 0.018 
        soma_Kslowcr = 0.000725
        soma_iCcr    = 0.00003
        soma_kadcr   = 0.003
        soma_cancr   = 0.001
        v_initcr     = -70
        
        self.soma.insert('Nafcr')
        self.soma.insert('kdrcr')  
        self.soma.insert('IKscr') 
        self.soma.insert('iCcr')    
        self.soma.insert('kadcr') 
        self.soma.insert('cancr') 
        self.soma.insert('cadyn')
        for sec in self.soma:
            sec.Nafcr.gnafbar = soma_nafcr
            sec.kdrcr.gkdrbar = soma_kdrcr
            sec.IKscr.gKsbar = soma_Kslowcr
            sec.iCcr.gkcbar = soma_iCcr
            sec.kadcr.gkabar = soma_kadcr
            sec.cancr.gcabar = soma_cancr
           
        self.soma.insert('pas')
        self.soma.cm = 1.2
        self.soma.g_pas = 1./20000.
        self.soma.e_pas = v_initcr
        self.soma.Ra = 150.
        
        for sec in self.all:
            name = sec.name().split('.')[-1]
            if name != 'soma':
                sec.insert('pas')
                sec.cm = 1.2
                sec.g_pas = 1./20000.
                sec.e_pas = v_initcr
                sec.Ra    = 150.
                
                sec.insert('Nafcr')
                sec.insert('kdrcr')
                
                for seg in sec:
                    seg.Nafcr.gnafbar = 0.018 * 5
                    seg.kdrcr.gkdrbar = 0.018* 0.5
                    


    def geom_nseg(self):
        lambda_f = h.lambda_f
        for seg in self.all:
            seg.nseg = int((seg.L/(0.1*lambda_f(100))+0.9)/2)*2+1
            
    def synapses(self):
        
        for sec in self.all:
            name = sec.name().split('.')[-1]
            self.synGroups['AMPA'][name] = {}
            self.synGroups['GABAA'][name] = {}
            self.synGroups['GABAB'][name] = {}
        