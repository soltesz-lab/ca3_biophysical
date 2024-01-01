COMMENT
	calcium accumulation into a volume of area*depth next to the
	membrane with a decay (time constant tau) to resting level
	given by the global calcium variable cai0_ca_ion
ENDCOMMENT

NEURON {
	THREADSAFE
	SUFFIX ccanl
	USEION nca READ ncai, inca, enca WRITE enca, ncai VALENCE 2
	USEION lca READ lcai, ilca, elca WRITE elca, lcai VALENCE 2
	USEION tca READ tcai, itca, etca WRITE etca, tcai VALENCE 2
	RANGE caiinf, catau, cai, ncai, lcai,tcai, eca, elca, enca, etca
}

UNITS {
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (milli/liter)
	(mA) = (milliamp)
	:FARADAY = 96520 (coul)  <--- This doesn't make its way onto the GPU, so we'll hard-code it.
	:R = 8.3134	(joule/degC)
}

PARAMETER {
	celsius = 6.3 (degC)
	depth = 200 (nm)	: assume volume = area*depth
	catau = 9 (ms)
	caiinf = 50.e-6 (mM)	: takes precedence over cai0_ca_ion
						: Do not forget to initialize in hoc if different
						: from this default.
	cao = 2 (mM)
	ica (mA/cm2)
	inca (mA/cm2)
	ilca (mA/cm2)
	itca (mA/cm2)
	cai= 50.e-6 (mM)
}

ASSIGNED {
	enca (mV)
	elca (mV)
	etca (mV)
	eca (mV)
}

STATE {
	ncai (mM)
	lcai (mM)
	tcai (mM)
}

INITIAL {
	ncai=caiinf/3
	lcai=caiinf/3
	tcai=caiinf/3
	cai = caiinf	
	eca = ktf() * log(cao/caiinf)	
	enca = eca
	elca = eca
	etca = eca
}


BREAKPOINT {
	SOLVE integrate METHOD cnexp
}

DERIVATIVE integrate {
	ncai' = -(inca)/depth/96520 * (1e7) + (caiinf/3 - ncai)/catau
	lcai' = -(ilca)/depth/96520 * (1e7) + (caiinf/3 - lcai)/catau
	tcai' = -(itca)/depth/96520 * (1e7) + (caiinf/3 - tcai)/catau
	cai = ncai+lcai+tcai	
	eca = ktf() * log(cao/cai)	
	enca = eca
	elca = eca
	etca = eca
}

FUNCTION ktf() (mV) {
	ktf = (1000)*8.3134*(celsius +273.15)/(2*96520)
} 
