: STDP by Hines, changed to dual exponential (BPG 6-1-09)
: Modified by BPG 13-12-08
: Limited weights: max weight is wmax and min weight is wmin
: (initial weight is specified by netconn - usually set to wmin)
: Rhythmic GABAB suppresses conductance and promotes plasticity.
: When GABAB is low, conductance is high and plasticity is off.

NEURON {
	POINT_PROCESS STDPE2
	RANGE tau1, tau2, e, i, d, p, dtau, ptau, thresh, wmax, wmin
	RANGE g, gbdel, gblen, gbint, gscale
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
	e = 0	(mV)
	wmax = 0 (uS)
	wmin = 0 (uS)	: not used - use netconn weight instead (BPG)
	d = 0 <0,1>: depression factor (multiplicative to prevent < 0)
	p = 0.5 : potentiation factor (additive, non-saturating)
	dtau = 34 (ms) : depression effectiveness time constant
	ptau = 17 (ms) : Bi & Poo (1998, 2001)
	thresh = 0 (mV)	: postsynaptic voltage threshold
	gbdel = 100 (ms) <1e-9,1e9> : initial GABAB off interval (ms)
	gbint = 100 (ms) <1e-9,1e9> : GABAB off interval (ms)
	gblen = 100 (ms) <1e-9,1e9> : GABAB on length (ms)
	gscale = 0.4	: relative suppression by GABAB
}

ASSIGNED {
	v (mV)
	i (nA)
	tpost (ms)
	on
	g (uS)
	gs
	factor
}

STATE {
	C (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	C = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
	gs=1
	on=1	: initially not plastic
	tpost = -1e9
	net_send(0, 1)
	net_send(gbdel, 3)	: initial GABAB off period
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - C
	i = g*gs*(v - e)
}

DERIVATIVE state {
	C' = -C/tau1
	B' = -B/tau2
}

NET_RECEIVE(w (uS), A, tpre (ms)) {
	INITIAL { A = 0  tpre = -1e9 }
	if (flag == 0) { : presynaptic spike  (after last post so depress)
:		printf("entry flag=%g t=%g w=%g A=%g tpre=%g tpost=%g\n", flag, t, w, A, tpre, tpost)
:		g = g + w + A	: only for single exp (BPG)
		C = C + (w + A)*factor
		B = B + (w + A)*factor
		tpre = t
		if (on == 1) {
			: A = A * (1 - d*exp((tpost - t)/dtau))
            : A = A + (wmax-w-A)*p*exp((tpost-t)/ptau)
            A = A + p*exp((tpost-t)/ptau)
            if (w + A >= wmax) {
               A = wmax - w
            }
		}
	}else if (flag == 2 && on == 1) { : postsynaptic spike
:		printf("entry flag=%g t=%g tpost=%g\n", flag, t, tpost)
		tpost = t
		FOR_NETCONS(w1, A1, tpre2) { : also can hide NET_RECEIVE args
:			printf("entry FOR_NETCONS w1=%g A1=%g tp=%g\n", w1, A1, tp)
			: A1 = A1 + (wmax-w1-A1)*p*exp((tpre2 - t)/ptau)
            A1 = A1 + p*exp((tpre2-t)/ptau)
            if (w1+A1 >= wmax) {
                A1 = wmax - w1
            }
		}
	} else if (flag == 1) { : flag == 1 from INITIAL block
:		printf("entry flag=%g t=%g\n", flag, t)
		WATCH (v > thresh) 2
	}

}