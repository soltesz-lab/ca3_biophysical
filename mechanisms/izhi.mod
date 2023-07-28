NEURON {
   POINT_PROCESS Izhi
   NONSPECIFIC_CURRENT Iizhi
   RANGE Iizhi
   RANGE k, vr, vt, a, b, vpeak, c, d, uinit
   RANGE Ifast
}

UNITS {
   (mV) = (millivolt)
        (nA) = (nanoamp)
   (pA) = (picoamp)
   (uS) = (microsiemens)
   (nS) = (nanosiemens)
}

PARAMETER {
   k = 0.7   (pA/mV2)
   vr = -60 (mV)
   vt = -40 (mV)
   a = 0.03 (1/ms)
   b = -2 (nS)
   vpeak = 35 (mV)
   c = -50 (mV)
   d = 100 (pA)
   uinit = 0 (pA)
}

ASSIGNED {
   v (mV)
   Iizhi (nA)
   Ifast (nA)
}

STATE {
   u (nA)
}

INITIAL {

   net_send(0,1)
}

BREAKPOINT {

   SOLVE recvars METHOD cnexp
   Ifast = k*(v-vr)*(v-vt)
   Iizhi = (-(Ifast - u))/1000
}

DERIVATIVE recvars {
   u' = a*(b*(v-vr)-u)
}

NET_RECEIVE (void) {
  if (flag == 1) {
    WATCH (v > vpeak) 2
  } else if (flag == 2) {
    net_event(t)
    v = c
    u = u+d
  }
}
