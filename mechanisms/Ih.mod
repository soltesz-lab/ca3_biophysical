COMMENT

Ih current	 - hyperpolarization-activated nonspecific Na and K channel
		 - contributes to the resting membrane potential
		 - controls the afterhyperpolarization
Reference:

1.	Maccaferri, G. and McBain, C.J. The hyperpolarization-activated current
	(Ih) and its contribution to pacemaker activity in rat CA1 hippocampal
	stratum oriens-alveus interneurons, J. Physiol. 497.1:119-130,
	1996.

		V1/2 = -84.1 mV
		   k = 10.2
		reversal potential = -32.9 +/- 1.1 mV

at -70 mV, currents were fitted by a single exponetial of t = 2.8+/- 0.76 s
at -120 mV, two exponentials were required, t1 = 186.3+/-33.6 ms 
t2 = 1.04+/-0.16 s


2.	Maccaferri, G. et al. Properties of the
	Hyperpoarization-activated current in rat hippocampal CA1 Pyramidal
	cells. J. Neurophysiol. Vol. 69 No. 6:2129-2136, 1993.

		V1/2 = -97.9 mV
		   k = 13.4
		reversal potential = -18.3 mV

3.	Pape, H.C.  Queer current and pacemaker: The
	hyperpolarization-activated cation current in neurons, Annu. Rev. 
	Physiol. 58:299-327, 1996.

		single channel conductance is around 1 pS
		average channel density is below 0.5 um-2
		0.5 pS/um2 = 0.00005 mho/cm2 = 0.05 umho/cm2		
4.	Magee, J.C. Dendritic Hyperpolarization-Activated Currents Modify
	the Integrative Properties of Hippocampal CA1 Pyramidal Neurons, J.
	Neurosci., 18(19):7613-7624, 1998

Deals with Ih in CA1 pyramidal cells.  Finds that conductance density
increases with distance from the soma.

soma g = 0.0013846 mho/cm2
dendrite g (300-350 um away) = 0.0125 mho/cm2
see Table 1 in th paper

ENDCOMMENT

 UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
}
 
NEURON {
        SUFFIX Ih
        USEION h READ eh WRITE ih VALENCE 1
        RANGE gkhbar,ih
        GLOBAL rinf, rexp, tau_r
}
 
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
 
PARAMETER {
        v (mV)
        p = 5 (degC)
        dt (ms)
        gkhbar = 0.001385 (mho/cm2)			
        eh = -32.9 (mV)
}
 
STATE {
        r
}
 
ASSIGNED {
        ih (mA/cm2)
	rinf rexp
	tau_r
}
 
BREAKPOINT {
        SOLVE deriv METHOD derivimplicit
        ih = gkhbar*r*(v - eh)
}
 
INITIAL {
	rates(v)
	r = rinf
}

DERIVATIVE deriv { :Computes state variable h at current v and dt.
	rates(v)
	r' = (rinf - r)/tau_r
}

PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        TABLE rinf, rexp, tau_r DEPEND dt, p FROM -200
TO 100 WITH 300
	rinf = 1/(1 + exp((v+84.1)/10.2))
	rexp = 1 - exp(-dt/(tau_r))
	tau_r = 100 + 1/(exp(-17.9-0.116*v)+exp(-1.84+0.09*v))
}
 
UNITSON

