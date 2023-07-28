TITLE HH channel that includes both a sodium and a delayed rectifier channel 
: and accounts for sodium conductance attenuation
: Bartlett Mel-modified Hodgkin - Huxley conductances (after Ojvind et al.)
: Terrence Brannon-added attenuation 
: Yiota Poirazi-modified Kdr and Na threshold and time constants to make it more stable
: Yiota Poirazi-modified threshold for soma/axon spike initiation (threshold about -57 mV),
: USC Los Angeles 2000, poirazi@LNC.usc.edu
: This file is used only in soma and axon sections


NEURON {
	SUFFIX hha2
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik
	NONSPECIFIC_CURRENT il
	RANGE gnabar, gkbar, gl, el
	RANGE ar2, vhalfs
	RANGE inf, fac, tau
	RANGE taus
	RANGE W
	GLOBAL taumin
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {                     :parameters that can be entered when function is called in cell-setup
 a0r = 0.0003 (ms)
        b0r = 0.0003 (ms)
        zetar = 12    
	zetas = 12   
        gmr = 0.2   
	ar2 = 1.0               :initialized parameter for location-dependent
                                :Na-conductance attenuation, "s", (ar=1 -> zero attenuation)
	taumin = 3   (ms)       :min activation time for "s" attenuation system
        vvs  = 2     (mV)       :slope for "s" attenuation system
        vhalfr = -60 (mV)       :half potential for "s" attenuation system
	W = 0.016    (/mV)      :this 1/61.5 mV
:	gnabar = 0.2 (mho/cm2)  :suggested conductance values
:	gkbar = 0.12 (mho/cm2)
:	gl = 0.0001  (mho/cm2)
        gnabar = 0   (mho/cm2)  :initialized conductances
	gkbar = 0    (mho/cm2)  :actual values set in cell-setup.hoc
	gl = 0       (mho/cm2)
	ena = 60     (mV)       :Na reversal potential (also reset in
	ek = -77     (mV)       :K reversal potential  cell-setup.hoc)
	el = -70.0   (mV)       :steady state 
	celsius = 34 (degC)
	v            (mV)
        dt
}

STATE {				:the unknown parameters to be solved in the DEs
	m h n s
}

ASSIGNED {			:parameters needed to solve DE
	ina (mA/cm2)
	ik (mA/cm2)
	il (mA/cm2)
	inf[4]
	fac[4]
	tau[4]
}

BREAKPOINT {
	SOLVE states
	ina = gnabar*m*m*h*s*(v - ena) :Sodium current
	ik = gkbar*n*n*(v - ek)        :Potassium current
	il = gl*(v - el)               :leak current
}

INITIAL {			:initialize the following parameter using states()
	states()
	s=1
	ina = gnabar*m*m*h*s*(v - ena)
	ik = gkbar*n*n*(v - ek)
	il = gl*(v - el)
}

PROCEDURE calcg() {
	mhn(v*1(/mV))
	m = m + fac[0]*(inf[0] - m)  :Na activation variable
	h = h + fac[1]*(inf[1] - h)  :Na inactivation variable
	n = n + fac[2]*(inf[2] - n)  :K activation variable
	s = s + fac[3]*(inf[3] - s)  :Na attenuation variable
}	

PROCEDURE states() {	: exact when v held constant
	calcg()
	VERBATIM
	return 0;
	ENDVERBATIM
}

FUNCTION varss(v, i) { :steady state values
	if (i==0) {
                varss = 1 / (1 + exp((v + 44)/(-3)))    :Na activation
 	}
	else if (i==1) {
                varss = 1 / (1 + exp((v + 49)/(3.5)))   :Na inactivation 
	}
	else if (i==2) {	
                varss = 1 / (1 + exp((v + 46.3)/(-3))) :K activation

	} else {
                :"s" activation system for spike attenuation - Migliore 96 model
		varss =     alpv(v,vhalfr)
       }
}

FUNCTION alpv(v(mV),vh) {    :used in "s" activation system infinity calculation
  alpv = (1+ar2*exp((v-vh)/vvs))/(1+exp((v-vh)/vvs))
}

FUNCTION alpr(v(mV)) {       :used in "s" activation system tau
  alpr = exp(1.e-3*zetar*(v-vhalfr)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betr(v(mV)) {       :used in "s" activation system tau
  betr = exp(1.e-3*zetar*gmr*(v-vhalfr)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION vartau(v, i) { :estimate tau values
	LOCAL tmp

	if (i==0) {
	    vartau = 0.05  :Na activation tau
	}
	else if (i==1) {
            vartau = 1     :Na inactivation tau
	}
	else if (i==2) {
            vartau = 3.5   :K activation
       	} else {
	     tmp = betr(v)/(a0r+b0r*alpr(v))
	     if (tmp<taumin) {tmp=taumin}
	VERBATIM
	ENDVERBATIM
	     vartau = tmp   :s activation tau
       }
}	

PROCEDURE mhn(v) {LOCAL a, b :rest = -70
:       TABLE infinity, tau, fac DEPEND dt, celsius FROM -100 TO 100 WITH 200
	FROM i=0 TO 3 {
		tau[i] = vartau(v,i)
		inf[i] = varss(v,i)
		fac[i] = (1 - exp(-dt/tau[i]))
	}
}















