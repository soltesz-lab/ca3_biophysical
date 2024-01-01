TITLE nca.mod  
 
COMMENT
konduktivitas valtozas hatasa- somaban 
ENDCOMMENT
 
UNITS {
	(mA) =(milliamp)
	(mV) =(millivolt)
	(uF) = (microfarad)
	(molar) = (1/liter)
	(nA) = (nanoamp)
	(mM) = (millimolar)
	(um) = (micron)
}
 
? interface 
NEURON { 
	THREADSAFE
	SUFFIX nca
	USEION nca READ enca WRITE inca VALENCE 2 
	RANGE  gnca
	RANGE gncabar
	RANGE cinf, ctau, dinf, dtau, inca
}
 
PARAMETER {
	v (mV) 
	celsius = 6.3 (degC)
	dt (ms) 
	gncabar (mho/cm2)
}
 
STATE {
	c
	d
}
 
ASSIGNED {
	gnca (mho/cm2)
	inca (mA/cm2)
	enca (mV)

	cinf dinf
	ctau (ms) dtau (ms) 
	cexp dexp      
} 

? currents
BREAKPOINT {
	SOLVE states METHOD cnexp
	gnca = gncabar*c*c*d
	inca = gnca*(v-enca)
}
 
UNITSOFF
 
INITIAL {
	trates(v)
	c = cinf
	d = dinf
}

? states
DERIVATIVE states {	:Computes state variables m, h, and n 
	trates(v)	:      at the current v and dt.
	c' = (cinf - c)/ctau
	d' = (dinf - d)/dtau
}
 
LOCAL q10

? rates
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL  alpha, beta, sum
	q10 = 3^((celsius - 6.3)/10)
                :"c" NCa activation system
	alpha = -0.19*vtrap(v-19.88,-10)
	beta = 0.046*exp(-v/20.73)
	sum = alpha+beta        
	ctau = 1/sum
	cinf = alpha/sum
                :"d" NCa inactivation system
	alpha = 0.00016*exp(-v/48.4)
	beta = 1/(exp((-v+39)/10)+1)
	sum = alpha+beta        
	dtau = 1/sum
	dinf = alpha/sum
}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL tinc
	TABLE  cinf, cexp, dinf, dexp, ctau, dtau
	DEPEND dt, celsius FROM -100 TO 100 WITH 200
                           
	rates(v)	: not consistently executed from here if usetable_hh == 1
		: so don't expect the tau values to be tracking along with
		: the inf values in hoc

	       tinc = -dt * q10
	cexp = 1 - exp(tinc/ctau)
	dexp = 1 - exp(tinc/dtau)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON

