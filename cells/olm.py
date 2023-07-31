' Author: Darian Hadjiabadi '

import numpy as np
from neuron import h, gui

class OLM(object):

    def __init__(self, gid):
        self.prelist   = []
        self.all        = None
        self.soma       = None
        self.dend1      = None
        self.dend2      = None
        self.axon       = None

        self.gid = gid
        self.synGroups = {}
        self.synGroups['AMPA'] = {}
        self.synGroups['GABAA'] = {}
        self.synGroups['GABAB'] = {} 
        self.internal_netcons = []
        self.external_netcons = {}

        self.init()
        
        self.spike_detector = h.NetCon(self.axon(0.5)._ref_v, None, sec=self.axon)
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
        self.dend1      = h.Section(name='dend1', cell=self)
        self.dend2      = h.Section(name='dend2', cell=self)
        self.axon       = h.Section(name='axon', cell=self)

        self.dend1.connect(self.soma(1))
        self.dend2.connect(self.soma(0))
        self.axon.connect(self.soma(0))
        
        #self.basic_shape()

    def basic_shape(self):
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0, 0, 0, 1, sec=self.soma)
        h.pt3dadd(15, 0, 0, 1, sec=self.soma)

        h.pt3dclear(sec=self.dend1)
        h.pt3dadd(15, 0, 0, 1, sec=self.dend1)
        h.pt3dadd(90, 0, 0, 1, sec=self.dend1)
 
        h.pt3dclear(sec=self.dend2)
        h.pt3dadd(0, 0, 0, 1, sec=self.dend2)
        h.pt3dadd(-74, 0, 0, 1, sec=self.dend2)
 
        h.pt3dclear(sec=self.axon)
        h.pt3dadd(15, 0, 0, 1, sec=self.axon)
        h.pt3dadd(15, 120, 0, 1, sec=self.axon)

    def subsets(self):
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def geom(self):
        self.soma.L = 20
        self.soma.diam = 10
 
        self.dend1.L = 250
        self.dend1.diam = 3

        self.dend2.L = 250
        self.dend2.diam = 3

        self.axon.L = 150
        self.axon.diam = 1.5

    def biophys(self):

        Rm = 20000

        for sec in self.all:
            sec.Ra = 150
            sec.cm = 1.3

        self.soma.insert('IA')
        for seg in self.soma:
            seg.IA.gkAbar = 0.0165
        self.soma.insert('Ih')
        for seg in self.soma:
            seg.Ih.gkhbar = 0.0005
        self.soma.insert('Ksoma')
        for seg in self.soma:
            seg.Ksoma.gksoma = 0.0319
        self.soma.insert('Nasoma')
        for seg in self.soma:
            seg.Nasoma.gnasoma = 0.0107
            seg.Nasoma.gl = 1. / Rm
            seg.Nasoma.el = -70

        self.dend1.insert('IA')
        for seg in self.dend1:
            seg.IA.gkAbar = 0.004
        self.dend1.insert('Kdend')
        for seg in self.dend1:
            seg.Kdend.gkdend = 2 * 0.023
        self.dend1.insert('Nadend')
        for seg in self.dend1:
            seg.Nadend.gnadend = 2 * 0.0117
            seg.Nadend.gl = 1. / Rm
            seg.Nadend.el = -70

        self.dend2.insert('IA')
        for seg in self.dend2:
            seg.IA.gkAbar = 0.004
        self.dend2.insert('Kdend')
        for seg in self.dend2:
            seg.Kdend.gkdend = 2 * 0.023
        self.dend2.insert('Nadend')
        for seg in self.dend2:
            seg.Nadend.gnadend = 2 * 0.0117
            seg.Nadend.gl = 1. / Rm
            seg.Nadend.el = -70
         
        self.axon.insert('Kaxon')
        for seg in self.axon:
            seg.Kaxon.gkaxon = 0.05104
        self.axon.insert('Naaxon')
        for seg in self.axon:
            seg.Naaxon.gnaaxon = 0.01712
            seg.Naaxon.gl = 1. / Rm
            seg.Naaxon.el = -70

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
        
 