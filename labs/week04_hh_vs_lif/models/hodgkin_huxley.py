# models/hodgkin_huxley.py
import numpy as np
from config import CM, GNA, GK, GL, ENA, EK, EL, V0, DT

# ---- Classic HH alpha/beta (V in mV, t in ms) ----
def alpha_m(V): return (0.1*(V+40.0)) / (1.0 - np.exp(-(V+40.0)/10.0))
def beta_m(V):  return 4.0*np.exp(-(V+65.0)/18.0)

def alpha_h(V): return 0.07*np.exp(-(V+65.0)/20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V+35.0)/10.0))

def alpha_n(V): return (0.01*(V+55.0)) / (1.0 - np.exp(-(V+55.0)/10.0))
def beta_n(V):  return 0.125*np.exp(-(V+65.0)/80.0)

def simulate_hh(I, t):
    V = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    n = np.zeros_like(t)

    V[0] = V0

    # Initialize gates to steady-state at V0
    m[0] = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
    h[0] = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
    n[0] = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

    for i in range(1, len(t)):
        v = V[i-1]

        # TODO 2-HH-1 (COMPLETED): Update gating variables (Euler)
        # Each gate x in {m, h, n} follows first-order kinetics:
        #     dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x
        # alpha_x is the opening rate (acts on the closed fraction 1 - x) and
        # beta_x is the closing rate (acts on the open fraction x).
        dm = alpha_m(v) * (1.0 - m[i-1]) - beta_m(v) * m[i-1]
        dh = alpha_h(v) * (1.0 - h[i-1]) - beta_h(v) * h[i-1]
        dn = alpha_n(v) * (1.0 - n[i-1]) - beta_n(v) * n[i-1]

        # TODO 3-HH-2 (COMPLETED): Compute ionic currents
        # Each current is conductance * driving force (V - E_ion).
        # Na+ needs three activation gates and one inactivation gate (m^3 h),
        # K+ needs four activation gates (n^4), leak is ohmic.
        INa = GNA * (m[i-1] ** 3) * h[i-1] * (v - ENA)
        IK  = GK  * (n[i-1] ** 4) * (v - EK)
        IL  = GL  * (v - EL)

        # TODO 4-HH-3 (COMPLETED): Update membrane voltage (Euler)
        # Cm dV/dt = I_inj - INa - IK - IL
        dV = (I[i-1] - INa - IK - IL) / CM

        # Update states (forward Euler step of size DT)
        m[i] = m[i-1] + DT * dm
        h[i] = h[i-1] + DT * dh
        n[i] = n[i-1] + DT * dn
        V[i] = V[i-1] + DT * dV

    return V, m, h, n