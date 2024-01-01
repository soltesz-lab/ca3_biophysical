TITLE l-calcium channel
: l-type calcium channel


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
	KTOMV = .0853 (mV/degC)
}

PARAMETER {
	v (mV)
	celsius 	(degC)
	glcabar		 (mho/cm2)
	ki=.001 (mM)
	cai (mM)
	cao (mM)
	tfa=1
}


NEURON {
    THREADSAFE
	SUFFIX lca
	USEION lca READ elca WRITE ilca VALENCE 2
	USEION ca READ cai, cao VALENCE 2 
	RANGE glcabar, cai, ilca, elca
	RANGE minf,matu
}

STATE {
	m
}

ASSIGNED {
	ilca (mA/cm2)
	glca (mho/cm2)
	minf
	matu   (ms)
	elca (mV)   

}

INITIAL {
	rate(v)
	m = minf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	glca = glcabar*m*m*h2(cai)
	ilca = glca*ghk(v,cai,cao)

}

FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}


FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
	LOCAL nu,f

	f = KTF(celsius)/2
	nu = v/f
	ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (DegC)) (mV) {
	KTF = ((25./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}

FUNCTION alp(v(mV)) (1/ms) {
	TABLE FROM -150 TO 150 WITH 200
	alp = 15.69*(-1.0*v+81.5)/(exp((-1.0*v+81.5)/10.0)-1.0)
}

FUNCTION bet(v(mV)) (1/ms) {
	TABLE FROM -150 TO 150 WITH 200
	bet = 0.29*exp(-v/10.86)
}

DERIVATIVE state {  
	rate(v)
	m' = (minf - m)/matu
}

PROCEDURE rate(v (mV)) {
	LOCAL a
	a = alp(v)
	matu = 1/(tfa*(a + bet(v)))
	minf = tfa*a*matu
}
 


