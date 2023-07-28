TITLE Dual-exponential model of NMDA receptors

COMMENT
Adapted from "Contribution of NMDA Receptor Channels to the Expression of LTP in the
Hippocampal Dentate Gyrus by Zhuo Wang," Dong Song, and Theodore W. Berger
Hippocampus, vol. 12, pp. 680-688, 2002.
ENDCOMMENT

NEURON {
	POINT_PROCESS Exp2NMDA_Wang
	NONSPECIFIC_CURRENT i
	RANGE tau1, tau2, e, i, Mg, eta, gamma, wf
	THREADSAFE
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
	(S)  = (siemens)
	(pS) = (picosiemens)
	(um) = (micron)
	(J)  = (joules)
}

PARAMETER {
: Parameters Control Neurotransmitter and Voltage-dependent gating of NMDAR
	tau1 = 0.6	(ms)
	tau2 = 55		(ms)
: Parameters Control voltage-dependent gating of NMDAR
	eta = 0.33	(/mM)
	gamma = 0.14	(/mV)
: Parameters Control Mg block of NMDAR
	Mg = 1		(mM)
	e = 0		(mV)
}

ASSIGNED {
	v		(mV)
	dt		(ms)
	i		(nA)
	factor
	wf
}

STATE {
	A
	B
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
	
	Mgblock(v)
}

BREAKPOINT {
	SOLVE state METHOD cnexp

	i = (B - A)*Mgblock(v)*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight) {
	wf = weight*factor
	A = A + wf
	B = B + wf
}

FUNCTION Mgblock(v(mV)) {
	: from Wang et. al 2002
	Mgblock = 1 / (1 + eta * Mg * exp( - (gamma * v)))
}
