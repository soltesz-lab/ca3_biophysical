TITLE CaGk
: Calcium activated K channel.
: Modified from Moczydlowski and Latorre (1983) J. Gen. Physiol. 82

UNITS {
	(molar) = (1/liter)
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)
}


NEURON {
	THREADSAFE
	SUFFIX cagk
	USEION nca READ ncai VALENCE 2
	USEION lca READ lcai VALENCE 2
	USEION tca READ tcai VALENCE 2
	USEION k READ ek WRITE ik
	RANGE gkbar,gkca, ik, oinf, otau, cai
}

UNITS {
	:FARADAY = (faraday)  (kilocoulombs) :<--Doesn't translate to GPU - use 96.485309 instead
	:R = 8.313424 (joule/degC)
}

PARAMETER {
	celsius		(degC)
	v		(mV)
	gkbar=.01	(mho/cm2)	: Maximum Permeability
	cai = 5.e-5	(mM)
	ek		(mV)

	d1 = .84
	d2 = 1.
	k1 = .48e-3	(mM)
	k2 = .13e-6	(mM)
	abar = .28	(/ms)
	bbar = .48	(/ms)
	st=1            (1)
	lcai		(mV)
	ncai		(mV)
	tcai		(mV)
}

ASSIGNED {
	ik		(mA/cm2)
	oinf
	otau		(ms)
	gkca          (mho/cm2)
}

INITIAL {
	cai= ncai + lcai + tcai
	rate(v,cai)
	o=oinf
}

STATE {
	o	: fraction of open channels
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gkca = gkbar*o^st
	ik = gkca*(v - ek)
}

DERIVATIVE state {	: exact when v held constant; integrates over dt step
	cai= ncai + lcai + tcai
	rate(v, cai)
	o' = (oinf - o)/otau
}

FUNCTION alp(v (mV), c (mM)) (1/ms) {
	alp = c*abar/(c + exp1(k1,d1,v))
}

FUNCTION bet(v (mV), c (mM)) (1/ms) {
	bet = bbar/(1 + c/exp1(k2,d2,v))
}

FUNCTION exp1(k (mM), d, v (mV)) (mM) {
	exp1 = k*exp(-2*d*96.485309*v/8.313424/(273.15 + celsius))
}

PROCEDURE rate(v (mV), c (mM)) {
	LOCAL a
	a = alp(v,c)
	otau = 1/(a + bet(v, c))
	oinf = a*otau
}

