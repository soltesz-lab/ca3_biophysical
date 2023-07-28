import numpy as np
import sys
import pickle
from pprint import pprint
from neuron import h
from cell_dhh import ca3cell

import matplotlib.pyplot as plt

netstims, netcons = [], []
syn_params = {}
syn_params['wmax'] = 5e-3
syn_params['plasticity'] = int(sys.argv[1])
rnd_count = 0

h.load_file("stdrun.hoc")
cellA = ca3cell(0, 'B', "CA3_Bakerb_marascoProp.pickle", syn_params)
cellB = ca3cell(1, 'B', "CA3_Bakerb_marascoProp.pickle", syn_params)
cellC = ca3cell(2, 'B', "CA3_Bakerb_marascoProp.pickle", syn_params)


def gen_netstims(n, cell, scaler=1):
    seed_start = np.random.randint(100,999) * scaler
    for i in range(n):
        ns = h.NetStim()
        ns.start = 250.
        ns.number = 300
        ns.interval = 30.
        ns.noise = 1.0
        ns.seed(seed_start + i)
        netstims.append(ns)

        nc = h.NetCon(ns, cell.synGroups['AMPANMDA']['lacunosumMEC'][0])
        nc.delay = 1.
        nc.weight[0] = 1.0e-3
        nc.weight[1] = 1.5e-3
        netcons.append(nc)


gen_netstims(50, cellA, scaler=1)
gen_netstims(50, cellB, scaler=10)
gen_netstims(50, cellC, scaler=100)

# netconAB = h.NetCon(cellA.axon(0.5)._ref_v, cellB.synGroups['AMPANMDA']['radiatum'][0], sec=cellA.axon)
# netconAB.delay = 1.
# netconAB.threshold = 0.0
# netconAB.weight[0] = 5.0e-4
# netconAB.weight[1] = 7.5e-4

# netconBA = h.NetCon(cellB.axon(0.5)._ref_v, cellA.synGroups['AMPANMDA']['radiatum'][0], sec=cellB.axon)
# netconBA.delay = 1.
# netconBA.threshold = 0.0
# netconBA.weight[0] = 5.0e-4
# netconBA.weight[1] = 7.5e-4

# netconBC = h.NetCon(cellB.axon(0.5)._ref_v, cellC.synGroups['AMPANMDA']['radiatum'][0], sec=cellB.axon)
# netconBC.delay = 1.
# netconBC.threshold = 0.0
# netconBC.weight[0] = 5.0e-4
# netconBC.weight[1] = 7.5e-4

# netconCB = h.NetCon(cellC.axon(0.5)._ref_v, cellB.synGroups['AMPANMDA']['radiatum'][0], sec=cellC.axon)
# netconCB.delay = 1.
# netconCB.threshold = 0.0
# netconCB.weight[0] = 5.0e-4
# netconCB.weight[1] = 7.5e-4

# netconCA = h.NetCon(cellC.axon(0.5)._ref_v, cellA.synGroups['AMPANMDA']['radiatum'][0], sec=cellC.axon)
# netconCA.delay = 1.
# netconCA.threshold = 0.0
# netconCA.weight[0] = 5.0e-4
# netconCA.weight[1] = 7.5e-4

# netconAC = h.NetCon(cellA.axon(0.5)._ref_v, cellC.synGroups['AMPANMDA']['radiatum'][0], sec=cellA.axon)
# netconAC.delay = 1.
# netconAC.threshold = 0.0
# netconAC.weight[0] = 5.0e-4
# netconAC.weight[1] = 7.5e-4




v_vecA = h.Vector()             # Membrane potential vector
v_vecB = h.Vector()
v_vecC = h.Vector()
t_vec = h.Vector()             # Time stamp vector
v_vecA.record(cellA.soma(0.5)._ref_v)
v_vecB.record(cellB.soma(0.5)._ref_v)
v_vecC.record(cellC.soma(0.5)._ref_v)
t_vec.record(h._ref_t)
h.tstop = 1000.0
h.dt    = 0.25
h.run()

plt.figure()
plt.plot(t_vec, v_vecA, color='k', label='cell A')
plt.plot(t_vec, v_vecB, color='r', label='cell B')
plt.plot(t_vec, v_vecC, color='g', label='cell C')
plt.legend()
plt.show()
