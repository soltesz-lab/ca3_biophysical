: simple first-order model of calcium dynamics

NEURON {
	SUFFIX cadyn
	USEION ca READ cai,ica WRITE cai 
	RANGE ca 
	RANGE depth,cainf,taur 
         
}

UNITS {
	(molar) = (1/liter)
	(mM)    = (milli/liter)
	(um)    = (micron)
	(mA)    = (milliamp)
	(msM)   = (ms mM)
	FARADAY = (faraday) (coul)
}

PARAMETER {
	depth = .1	  (um)
	taur  = 200   (ms)
	cainf = 50e-6 (mM)
	cai		      (mM)
}

ASSIGNED {
	ica		        (mA/cm2)
	drive_channel	(mM/ms)
}

STATE {
	ca (mM)
}

BREAKPOINT {
	SOLVE state METHOD euler
}

DERIVATIVE state { 

	drive_channel =  - (10000) * ica / (2 * FARADAY * depth)
	if (drive_channel <= 0.) { drive_channel = 0.  }
    
    ca' = drive_channel/18 + (cainf - ca)/taur*11
	cai = ca
}


INITIAL {
	ca = cainf
}
