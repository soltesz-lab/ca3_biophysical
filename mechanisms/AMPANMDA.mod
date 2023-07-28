TITLE Dual-exponential Exp2Syn with added dual-exponential model of NMDA receptors

COMMENT
Written by Darian Hadjiabadi
AMPA rececptor adapted from h.Exp2Syn
NMDA receptor Adapted from "Contribution of NMDA Receptor Channels to the Expression of LTP in the
Hippocampal Dentate Gyrus by Zhuo Wang," Dong Song, and Theodore W. Berger
Hippocampus, vol. 12, pp. 680-688, 2002.
ENDCOMMENT

NEURON {
    POINT_PROCESS AMPANMDA
    NONSPECIFIC_CURRENT i
    RANGE gid, tau1ampa, tau2ampa, tau1nmda, tau2nmda, e, i, Mg, eta, gamma, wfampa, wfnmda, thresh
    RANGE d, p, dM, dV, ptau, wmax, wmin, pon
    THREADSAFE
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
    (S)  = (siemens)
    (pS) = (picosiemens)
    (um) = (micron)
    (J)  = (joules)
}

PARAMETER {
    gid = 0
    e = 0 (mV)
: Parameters control AMPA gating
    tau1ampa = 0.1 (ms)
    tau2ampa = 10.0 (ms)
    
: Parameters Control Neurotransmitter and Voltage-dependent gating of NMDAR
    tau1nmda = 0.6  (ms)
    tau2nmda = 55  (ms)
: Parameters Control voltage-dependent gating of NMDAR
    eta = 0.33 (/mM)
    gamma = 0.14 (/mV)
: Parameters Control Mg block of NMDAR
    Mg = 1  (mM)
: Spike threshold for calculating plasticity changes
    thresh = 0 (mV)
: plasticity related parameters
    d  = 8.
    p  = 1.2
    dM = -22 (ms)
    dV = 5 (ms)
    ptau = 10 (ms)
    pi = 3.14159
    wmax = 0.005
    wmin = 0.00001
    pon  = 1
}

ASSIGNED {
    v (mV)
    dt (ms)
    i (nA)
    tpost (ms)
    wfampa
    factorampa
    wfnmda
    factornmda
    
}

STATE {
    Aampa (uS)
    Bampa (uS)
    Anmda (uS)
    Bnmda (uS)
}

INITIAL {
    LOCAL tpampa, tpnmda
    if (tau1ampa/tau2ampa > .9999) {
        tau1ampa = .9999*tau2ampa
    }
    Aampa = 0
    Bampa = 0
    tpampa = (tau1ampa*tau2ampa)/(tau2ampa - tau1ampa) * log(tau2ampa/tau1ampa)
    factorampa = -exp(-tpampa/tau1ampa) + exp(-tpampa/tau2ampa)
    factorampa = 1/factorampa
    
    if (tau1nmda/tau2nmda > .9999) {
        tau1nmda = 0.9999*tau2nmda
    }
    Anmda = 0
    Bnmda = 0
    tpnmda = (tau1nmda*tau2nmda)/(tau2nmda - tau1nmda) * log(tau2nmda/tau1nmda)
    factornmda = -exp(-tpnmda/tau1nmda) + exp(-tpnmda/tau2nmda)
    factornmda = 1/factornmda
    Mgblock(v)
    
    tpost = -1e9
    net_send(0, 1)
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = (Bampa - Aampa)*(v-e) + (Bnmda - Anmda)*Mgblock(v)*(v-e)
}

DERIVATIVE state {
    Aampa' = -Aampa/tau1ampa
    Bampa' = -Bampa/tau2ampa
    
    Anmda' = -Anmda/tau1nmda
    Bnmda' = -Bnmda/tau2nmda
}

NET_RECEIVE(weightampa, weightnmda, A, tpre (ms)) {
    INITIAL { 
        weightnmda = weightnmda
        A = 0
        tpre = -1e9
    }
    if (flag == 0) { : presynpatic spike detection
        if (pon == 1) {
            :printf("gid: %g. presynaptic spike time: %g. ampa weight: %g\. nmda weight: %g. \n", gid, t, (weightampa+A), weightnmda)
            wfampa = (weightampa+A)*factorampa       
            tpre = t
            A = A * (1-(d*exp(-((tpost-t)-dM)^2/(2*dV*dV))) /(sqrt(2*pi)*dV))
        } 
        else {
            wfampa = weightampa*factorampa
        }
        Aampa = Aampa + wfampa
        Bampa = Bampa + wfampa

        wfnmda = weightnmda*factornmda
        Anmda = Anmda + wfnmda
        Bnmda = Bnmda + wfnmda

    } 
    else if (flag == 1) { : from INITIAL block, will detect postsynaptic spike if voltage > thresh and send flag = 2
        WATCH (v > thresh) 2
    } 
    
    else if (flag == 2) { : post synaptic spike detection
        if (pon == 1) {
            :printf("gid: %g postsynaptic spike detected\n", gid)
            tpost = t
            FOR_NETCONS(w1, w2, A1, tpr) { : also can hide NET_RECEIVE args
                :printf("netcon loop gid: %g. tpre-t: %g\n", gid, (tpr-t))
                :gitprintf("post-pre: %g. andd %g\n",tpost-tpr, wmax-w1-A1)
                A1 = A1 + (wmax-w1-A1)*p*exp((tpr - t)/ptau)
            }
        }
    }
}

FUNCTION Mgblock(v(mV)) {
    : from Wang et. al 2002
    Mgblock = 1 / (1 + eta * Mg * exp( - (gamma * v)))
}