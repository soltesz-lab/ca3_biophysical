TITLE  Na persistent channel
: used in distal oblique dendrites to assist Ca spike initiation  
: a typo in the exponential function was pointed out by Michele Migliore and
: corrected by Yiota Poirazi on December 4th, 2003
NEURON {
	SUFFIX nap
	USEION na READ ena WRITE ina
        RANGE  gnabar,vhalf, K

}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

INDEPENDENT {t FROM 0 TO 1 WITH 100 (ms)}

PARAMETER {                         : parameters that can be entered when function is called in cell-setup 
        dt (ms) 
	v (mV)
        ena = 50           (mV)     : Na reversal potential  (reset in cell-setup.hoc)
	K = 4.5            (1)      : slope of steady state variable
:	gnabar = 0.001e-2 (mho/cm2) : suggested conductance, 1 percent of the transient Na current
	gnabar = 0                  : initialized conductance
	vhalf  = -50.4    (mV)      : half potential
}	

STATE { n }

ASSIGNED {
	ina (mA/cm2)
}

INITIAL {
:	SOLVE states
}

BREAKPOINT {
	SOLVE states
	ina = gnabar*n*n*n*(v-ena)
}

PROCEDURE states() {     : exact when v held constant; integrates over dt step

:        n = 1 / (1 + (exp(vhalf - v))/K) : steady state value !!!typo in the exponential function!!!

        n = 1 / (1 + (exp(vhalf - v)/K)) : steady state value
        VERBATIM
        return 0;
        ENDVERBATIM
}















