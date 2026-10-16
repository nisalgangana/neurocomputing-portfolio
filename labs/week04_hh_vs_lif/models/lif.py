# models/lif.py
import numpy as np
from config import TAU_M, V_REST, V_RESET, V_TH, R_M, DT, REFRAC_MS

def simulate_lif(I, t):
    """
    LIF: tau dV/dt = -(V - Vrest) + R*I
    Spike if V >= Vth -> reset and refractory
    """
    V = np.zeros_like(t)
    V[0] = V_REST

    spike_times = []
    refrac_steps = 0

    for i in range(1, len(t)):

        # If in refractory, hold at reset
        if refrac_steps > 0:
            V[i] = V_RESET
            refrac_steps -= 1
            continue

        dV = (-(V[i-1] - V_REST) + R_M * I[i-1]) / TAU_M
        V[i] = V[i-1] + DT * dV

        if V[i] >= V_TH:
            spike_times.append(t[i])
            V[i] = V_RESET

            # TODO 1-LIF-1 (COMPLETED):
            # Implement refractory by converting REFRAC_MS to simulation steps.
            # REFRAC_MS is a duration in ms and DT is the length of one
            # simulation step in ms, so the refractory period lasts
            # REFRAC_MS / DT steps. The counter is decremented at the top of the
            # loop, where V is held at V_RESET, so the voltage stays clamped at
            # the reset value for the whole refractory window.
            refrac_steps = int(round(REFRAC_MS / DT))

    return V, np.array(spike_times)