from neuron import h

class Septal(object):
    
    def __init__(self, gid, bursting_params):
        self.gid = gid
        self.bursting_params = bursting_params
        self.soma  = h.Section(name='soma', cell=self)
        self.bcell = h.BurstStim2(sec=self.soma)
        
        for param in self.bursting_params.keys():
            setattr(self.bcell, param, self.bursting_params[param])

            
        self.spike_detector = h.NetCon(self.bcell, None)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)