TITLE ...just to store peak membrane voltage
: M.Migliore June 2001

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {
	v (mV)
}


NEURON {
	SUFFIX ds_CA3
        RANGE vmax
}

ASSIGNED {
	vmax
}

INITIAL {
	vmax=v
}


BREAKPOINT {
	if (v>vmax) {vmax=v}
}
