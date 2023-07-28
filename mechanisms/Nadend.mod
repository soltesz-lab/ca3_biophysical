COMMENT

Sodium current for the dendrites

References:

1.	Martina, M., Vida, I., and Jonas, P.  Distal initiation and active
	propagation of action potentials in interneuron dendrites,
	Science, 287:295-300, 2000.

			soma	axon-lacking dend	axon-bearing dend
Na+	gmax	    107 ps/um2	   117 ps/um2		   107 ps/um2
	slope 	    10.9 mV/e	   11.2 mV/e		   11.2 mV/e
	V1/2        -37.8 mV       -45.6 mV                -45.6 mV



2.	Marina, M. and Jonas, P.  Functional differences in Na+ channel
	gating between fast-spiking interneurons and principal neurones of rat
	hippocampus, J. Physiol., 505.3:593-603, 1997.

*Note* The interneurons here are basket cells from the dentate gyrus.

Na+	Activation V1/2				-25.1 mV
	slope			 		11.5
	Activation t (-20 mV)	 		0.16 ms
	Deactivation t (-40 mV)	 		0.13 ms
 	Inactivation V1/2			-58.3 mV
	slope			 		6.7
	onset of inactivation t (-20 mV)	1.34 ms
	onset of inactivation t (-55 mV)	18.6 ms
	recovery from inactivation t		2.0 ms
	(30 ms conditioning pulse)
	recovery from inactivation t		2.7 ms
	(300 ms conditioning pulse)

ENDCOMMENT
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
}
 
NEURON {
        SUFFIX Nadend
        USEION na READ ena WRITE ina
        NONSPECIFIC_CURRENT il
        RANGE gnadend, gl, el, ina
        GLOBAL minf, hinf, hexp, mtau, htau
}
 
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
 
PARAMETER {
        v (mV)
        celsius = 24 (degC)
        dt (ms)
        gnadend = .0117 (mho/cm2)
        ena = 90 (mV)
        gl = .00005 (mho/cm2)
        el = -70 (mV)
}
 
STATE {
        m h 
}
 
ASSIGNED {
        ina (mA/cm2)
        il (mA/cm2)
        minf 
	mexp 
	hinf 
	hexp
	mtau (ms)
	htau (ms)
}
 
INITIAL {
	m = minf
	h = hinf
}

BREAKPOINT {
        SOLVE states
	ina = gnadend*minf*minf*minf*h*(v - ena)    
        il = gl*(v - el)
}

PROCEDURE states() {	:exact when v held constant
	evaluate_fct(v)
	h = h + hexp*(hinf - h)
	VERBATIM
	return 0;
	ENDVERBATIM 
}
UNITSOFF

PROCEDURE evaluate_fct(v(mV)) {  :Computes rate and other constants at 
		      :current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL q10, tinc, alpha, beta
        TABLE minf, hinf, hexp, mtau, htau DEPEND dt, celsius FROM -200 TO 
100 WITH 300
:		q10 = 3^((celsius - 24)/10)
		q10 = 1	: BPG
		tinc = -dt*q10
		alpha = 0.1*vtrap(-(v+45),10)
		beta = 4*exp(-(v+70)/18)
		mtau = 1/(alpha + beta)
		minf = alpha*mtau
		alpha = 0.07*Exp(-(v+70)/20)
		beta = 1/(1+Exp(-(v+40)/10))
		htau = 1/(alpha + beta)
		hinf = alpha*htau
		hexp = 1-Exp(tinc/htau)
}
FUNCTION vtrap(x,y) {	:Traps for 0 in denominator of rate eqns.
		if (fabs(x/y) < 1e-6) {
			vtrap = y*(1 - x/y/2)
		}else{
			vtrap = x/(Exp(x/y) - 1)
		}
}
FUNCTION Exp(x) {
		if (x < -100) {
			Exp = 0
		}else{
			Exp = exp(x)
		}
}
UNITSON
