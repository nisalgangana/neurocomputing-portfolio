# config.py  (units: ms, mV, uA/cm^2 for HH standard)

# Time
DT = 0.01          # ms  (HH needs small dt for stability)
T_TOTAL = 80.0     # ms

# ----- LIF parameters (simple) -----
TAU_M = 20.0       # ms
V_REST = -65.0     # mV
V_RESET = -65.0    # mV
V_TH = -50.0       # mV
R_M = 10.0         # "resistance" scaling (arbitrary units)
REFRAC_MS = 5.0    # ms

# ----- HH parameters (classic squid axon) -----
CM = 1.0           # uF/cm^2
GNA = 120.0        # mS/cm^2
GK  = 36.0         # mS/cm^2
GL  = 0.3          # mS/cm^2
ENA = 50.0         # mV
EK  = -77.0        # mV
EL  = -54.4        # mV
V0  = -65.0        # mV initial membrane potential