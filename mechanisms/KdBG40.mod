TITLE Kd current

COMMENT 
		For act & inact tau, 
		  Storm JF (1988) Nature, 336:379-381
		For vhalf and slope of xinf and yinf,
		  BossuGahwiler1_JP96.pdf &  TytgatDaenens_BJP97(kv11).pdf 
		To fit simulation to the CA3 data, LSH changed
			tauy, 150 -> 100 ms (Hyun et al JP 2013)
			vhalfx, -55 -> -48(Saviane JP 2003 and Hyun JP 2013)
			vhalfy, -88 -> -90 (Saviane JP 2003)
			Ky, 0.6e-3 -> 1e-3
			zettax, 2 -> 2.5 (Hyun JP 2013)
ENDCOMMENT

NEURON {
	SUFFIX KdBG
	USEION k WRITE ik
	RANGE  gbar,ik
	GLOBAL xtau, ytau, xinf, yinf
}

UNITS {
	(S)	= (siemens)
	(mA)	= (milliamp)
	(mV)	= (millivolt)
	FARADAY	= 96480 (coulombs)
	R	= 8.314  (joule/degC)
}

PARAMETER {
	v		(mV)
	gbar	= 1.0e-3	(S/cm2)
	celsius	= 25	(degC)
	Kx = 1   (1/ms)
	Ky	=   1e-3	(1/ms)
	zettax	=  2.5		(1)
	zettay	=  -1.5		(1)
	vhalfx	= -48.0		(mV)
	vhalfy	= -90.0		(mV)
	taux	=   1		(ms)
	tauy	=   100		(ms)
	q10	= 1.0	(1)    : no temp dependence
	FRT = 39 (coulombs/joule) 
}

ASSIGNED {
	ik     	(mA/cm2)
	xtau    (ms)
	ytau    (ms)
	xinf	(1)
	yinf	(1)
}

STATE { xs ys }

BREAKPOINT { 
	SOLVE states METHOD cnexp
	ik= gbar * xs * ys * ( v + 90.0 ) 
}

DERIVATIVE states {
	rates(v)
	xs'= (xinf- xs)/ xtau	
	ys'= (yinf- ys)/ ytau
}

INITIAL {
	rates(v)
	xs= xinf
	ys= yinf
}

PROCEDURE rates(v (mV)) { LOCAL a, b, T, qt
	T = celsius + 273.15  
	qt = q10 ^( (celsius-35.0) / 10.0(K) )
	a = qt*Kx*exp( (1.0e-3)*  zettax*(v-vhalfx)*FRT )
	b = qt*Kx*exp( (1.0e-3)* -zettax*(v-vhalfx)*FRT )
	xinf = a / ( a + b )
	xtau = 1 /(a + b)+ taux

	a = qt*Ky*exp( (1.0e-3)*  zettay* (v-vhalfy)*FRT )
	b = qt*Ky*exp( (1.0e-3)* -zettay* (v-vhalfy)*FRT )
	yinf = a   / ( a + b )
	ytau = 1.0 / ( a + b ) + tauy
}






