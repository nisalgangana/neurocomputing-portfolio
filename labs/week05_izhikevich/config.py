# config.py

DT = 0.1          # ms
T_TOTAL = 1000    # ms

# Default input current
I_CONST = 10

# Izhikevich neuron presets
REGULAR_SPIKING = {"a": 0.02, "b": 0.2, "c": -65, "d": 8}
FAST_SPIKING    = {"a": 0.1,  "b": 0.2, "c": -65, "d": 2}
BURSTING        = {"a": 0.02, "b": 0.2, "c": -50, "d": 2}