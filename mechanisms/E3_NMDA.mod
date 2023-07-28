: STDP by Hines, changed to dual exponential (BPG 6-1-09)
: Modified by BPG 13-12-08
: Limited weights: max weight is wmax and min weight is wmin
: (initial weight is specified by netconn - usually set to wmin)
: Rhythmic GABAB suppresses conductance and promotes plasticity.
: When GABAB is low, conductance is high and plasticity is off.

NEURON {
	POINT_PROCESS E3_NMDA
	RANGE tau1, tau2, tau3, wtau2, wtau3, factor, e, i
	RANGE g, B, C, E
	RANGE wf
	RANGE eta, gamma, mgblock
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1=.1 (ms) <1e-9,1e9>
	tau2 = 10 (ms) <1e-9,1e9>
	tau3 = 11 (ms) <1e-9,1e9>
	e = 0	(mV)
	factor = 10                 : time to peak - needs to be solved outside nmodl
	wtau2 = 0.5             : weighting for second exponential
	
: Parameters Control Mg block of NMDAR
	Mg = 1	(mM)
	:scaleFactor = 3.25  : scaling factor for IV curve
}

ASSIGNED {
	v (mV)
	i (nA)
	tpost (ms)
	g (uS)
	wf
	mgblock
	wtau3

}

STATE {
	C (uS)
	B (uS)
	E (uS)

}

INITIAL {
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	if (tau2/tau3 > .9999) {
	    tau2 = .9999*tau3
    }
    wtau3 = 1-wtau2
	C = 0
	B = 0
	E = 0
	tpost = -1e9
	net_send(0, 1)
	mgblock = Mgblock(v)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = wtau2*B + wtau3*E - C
	mgblock = Mgblock(v)
	i = g*(v - e)*mgblock : * scaleFactor

}

DERIVATIVE state {
	C' = -C/tau1
	B' = -B/tau2
	E' = -E/tau3
      
}

NET_RECEIVE(w (uS)) {
	if (flag == 0) { : presynaptic spike  (after last post so depress)	
		wf = w*factor
		C = C + wf
		B = B + wf
		E = E + wf
	}
}

FUNCTION Mgblock(v(mV)) {
	: from Wang et. al 2002
	Mgblock = 1 / (1 + exp(-62 * v * 0.001(1/mV)) * Mg / 3.57)
}
