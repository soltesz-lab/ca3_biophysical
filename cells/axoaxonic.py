' Author: Darian Hadjiabadi '

import numpy as np
from neuron import h, gui

class AAC(object):

    def __init__(self, gid):
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
    
        self.gid = gid
        self.synGroups = {}
        self.synGroups['AMPA'] = {}
        self.synGroups['GABAA'] = {}
        self.synGroups['GABAB'] = {}
        self.internal_netcons = []
        self.external_netcons = {}

        self.init()
        
        self.spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)
        
        self.spike_threshold = -45.0
        
        
    def init(self):
        self.topol()
        self.subsets()
        self.geom()
        self.biophys()
        self.geom_nseg()
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

        self.radT2.connect(self.soma(1))
        self.radM2.connect(self.radT2(1))
        self.radt2.connect(self.radM2(1))
        self.lmM2.connect(self.radt2(1))
        self.lmt2.connect(self.lmM2(1))
        self.radT1.connect(self.soma(0))
        self.radM1.connect(self.radT1(1))
        self.radt1.connect(self.radM1(1))
        self.lmM1.connect(self.radt1(1))
        self.lmt1.connect(self.lmM1(1))
        self.oriT1.connect(self.soma(0))
        self.oriM1.connect(self.oriT1(1))
        self.orit1.connect(self.oriM1(1))
        self.oriT2.connect(self.soma(1))
        self.oriM2.connect(self.oriT2(1))
        self.orit2.connect(self.oriM2(1))
        
        self.basic_shape()

    def basic_shape(self):
        pass

    def subsets(self):
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def geom(self):
        self.soma.L = 20
        self.soma.diam = 10

        self.radT2.L = 100
        self.radT2.diam = 4
 
        self.radM2.L = 100
        self.radM2.diam = 3
 
        self.radt2.L = 200
        self.radt2.diam = 2

        self.lmM2.L = 100
        self.lmM2.diam = 1.5
 
        self.lmt2.L = 100
        self.lmt2.diam = 1
 
        self.radT1.L = 100
        self.radT1.diam = 4

        self.radM1.L = 100
        self.radM1.diam = 3

        self.radt1.L = 200
        self.radt1.diam = 2

        self.lmM1.L = 100
        self.lmM1.diam = 1.5

        self.lmt1.L = 100
        self.lmt1.diam = 1

        self.oriT1.L = 100
        self.oriT1.diam = 2

        self.oriM1.L = 100
        self.oriM1.diam = 1.5

        self.orit1.L = 100
        self.orit1.diam = 1

        self.oriT2.L = 100
        self.oriT2.diam = 2

        self.oriM2.L = 100
        self.oriM2.diam = 1.5

        self.orit2.L = 100
        self.orit2.diam = 1
 


    def biophys(self):

        gna = 0.15
        for sec in self.all:
            sec.insert('ichan2')
            for seg in sec:
                seg.ichan2.gnatbar = gna
                seg.ichan2.gkfbar = 0.013
                seg.ichan2.gl = 0.00018
                seg.ichan2.el = -60

            sec.insert('ccanl')
            for seg in sec:
                seg.ccanl.catau = 10
                seg.ccanl.caiinf = 5.0e-6
 
            sec.insert('borgka')
            for seg in sec:
                seg.borgka.gkabar = 0.00015

            sec.insert('nca')
            for seg in sec:
                seg.nca.gncabar = 0.0008

            sec.insert('lca')
            for seg in sec:
                seg.lca.glcabar = 0.005

            sec.insert('gskch')
            for seg in sec:
                seg.gskch.gskbar = 0.000002

            sec.insert('mykca')
            for seg in sec:
                seg.mykca.gkbar = 0.0002
        

            sec.cm = 1.4 
            sec.Ra = 100
            sec.enat = 55
            sec.ekf = -90
            sec.ek = -90
            sec.elca = 130
            sec.enca = 130
            sec.esk = -90     
         

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
     
        
#         syn_ = h.MyExp2Syn(self.soma(0.5))
#         syn_.tau1 = 2.0
#         syn_.tau2 = 6.3
#         syn_.e = 0
#         self.prelist.append(syn_)
      
        
#         syn_ = h.MyExp2Syn(self.soma(0.5))
#         syn_.tau1 = 0.180
#         syn_.tau2 = 0.45
#         syn_.e = -75
#         self.prelist.append(syn_)
        
#         syn_ = h.MyExp2Syn(self.lmM1(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)        

#         syn_ = h.MyExp2Syn(self.lmM2(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.radM1(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.radM2(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.radT1(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.radT2(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.oriT1(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)
        

#         syn_ = h.MyExp2Syn(self.oriT2(0.5))
#         syn_.tau1 = 0.5
#         syn_.tau2 = 3
#         syn_.e = 0
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.soma(0.5))
#         syn_.tau1 = 1
#         syn_.tau2 = 8
#         syn_.e = -75
#         self.prelist.append(syn_)
      
#         syn_ = h.MyExp2Syn(self.soma(0.6))
#         syn_.tau1 = 1
#         syn_.tau2 = 8
#         syn_.e = -75
#         self.prelist.append(syn_)
         
#         syn_ = h.MyExp2Syn(self.oriT1(0.6))
#         syn_.tau1 = 1
#         syn_.tau2 = 8
#         syn_.e = -75
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.oriT2(0.6))
#         syn_.tau1 = 1
#         syn_.tau2 = 8
#         syn_.e = -75
#         self.prelist.append(syn_)

#         syn_ = h.MyExp2Syn(self.oriT1(0.6))
#         syn_.tau1 = 35
#         syn_.tau2 = 100
#         syn_.e = -75
#         self.prelist.append(syn_)
        
#         syn_ = h.MyExp2Syn(self.oriT2(0.6))
#         syn_.tau1 = 35
#         syn_.tau2 = 100
#         syn_.e = -75
#         self.prelist.append(syn_)