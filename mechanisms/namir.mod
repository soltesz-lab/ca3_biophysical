TITLE  Na channel
: with recovery from inactivation and ir. M.Migliore BJ 1996
: modified to be used with cvode M.Migliore June 2001
NEURON {
	SUFFIX namr
	USEION na READ ena WRITE ina
	NONSPECIFIC_CURRENT Ir
        RANGE gnabar,b,vhalf,vhalfn,vhalfl,vvh
        GLOBAL ninf,linf,taul,taun,rinf,taur,inf
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {
        dt (ms)
	v (mV)
        ena (mV)
	celsius 	(degC)
	gnabar=.01 (mho/cm2)
	vhalf=-50    (mV)
	zeta=20       (1)
        vhalfn=-30   (mV)
        vhalfr=-60   (mV)
        vhalfl=-57   (mV)
        a0l=0.1      (/ms)
        a0n=2    (/ms)
        a0r=0.0003    (ms)
        zetan=-4    (1)
        zetar=12    (1)
        zetal=4    (1)
        gmn=0.9   (1)
        gmr=0.2   (1)
        gml=0.65   (1)
	lmax=1   (1)
	nmax=0.01   (1)
	rmin=3 (1)
	vvh=-58     (mv)
	vvs=2 	(1)
	b=0
}


STATE {
	n
        l
	r
	fr
}

ASSIGNED {
	Ir  (mA/cm2)
	ina (mA/cm2)
	inf
        ninf
        linf      
	rinf
        taul
        taun
	taur
}

INITIAL {
	rates(v)
	n=ninf
	l=linf
	r=rinf
	fr=inf

}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gnabar*n*l*r*(v-ena)
	Ir = -ina*fr
}

FUNCTION alp(v(mV)) { 
  alp = exp( 1.e-3*zeta*(v-vhalf)*9.648e4/(8.315*(273.16+celsius)))
}       

FUNCTION alpn(v(mV)) {
  alpn = exp(1.e-3*zetan*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betn(v(mV)) {
  betn = exp(1.e-3*zetan*gmn*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION alpv(v(mV)) {
         alpv = (1+b*exp((v-vvh)/vvs))/(1+exp((v-vvh)/vvs))
}

FUNCTION alpr(v(mV)) {
  alpr = exp(1.e-3*zetar*(v-vhalfr)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betr(v(mV)) {
  betr = exp(1.e-3*zetar*gmr*(v-vhalfr)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION alpl(v(mV)) {
  alpl = exp(1.e-3*zetal*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betl(v(mV)) {
  betl = exp(1.e-3*zetal*gml*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
}

DERIVATIVE states { 
        rates(v)
        n' = (ninf-n)/taun
        l' = (linf - l)/taul
        r' = (rinf - r)/taur
	fr = inf
}

PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a,q10
        q10=1.5^((celsius-22)/10)
        inf = 1/(1 + alp(v))
        a = alpn(v)
        ninf = 1/(1 + a)
        taun = betn(v)/(q10*a0n*(1+a))
	if (taun<nmax) {taun=nmax}
        a = alpl(v)
        linf = 1/(1+ a)
        taul = betl(v)/(q10*a0l*(1 + a))
	if (taul<lmax) {taul=lmax}
        rinf = alpv(v)
        taur = betr(v)/(a0r*(1+alpr(v)))
	if (taur<rmin) {taur=rmin}
}






