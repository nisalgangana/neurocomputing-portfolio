# models/izhikevich.py
import numpy as np
from config import DT

def simulate_izhikevich(I, t, a, b, c, d):
    """
    Simulate Izhikevich neuron model.
    
    Parameters:
        I : input current array
        t : time array
        a, b, c, d : model parameters
    
    Returns:
        v_trace : membrane potential trace
        u_trace : recovery variable trace
        spike_times : spike times
    """
    
    v_trace = np.zeros_like(t)
    u_trace = np.zeros_like(t)
    spike_times = []

    # Initial conditions
    v_trace[0] = -65
    u_trace[0] = b * v_trace[0]

    for i in range(1, len(t)):
        v = v_trace[i-1]
        u = u_trace[i-1]

        # TODO 1:
        # Compute dv and du using the Izhikevich equations
        dv = 0.04 * v**2 + 5 * v + 140 - u + I[i-1]
        du = a * (b * v - u)

        # Update state variables using Euler method
        v_trace[i] = v + DT * dv
        u_trace[i] = u + DT * du

        # TODO 2:
        # Implement spike reset:
        if v_trace[i] >= 30:
            spike_times.append(t[i])   # record the spike time (ms)
            v_trace[i] = c             # reset membrane potential
            u_trace[i] = u_trace[i] + d  # increment recovery variable

    return v_trace, u_trace, np.array(spike_times)