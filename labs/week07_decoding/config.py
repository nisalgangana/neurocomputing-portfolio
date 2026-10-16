import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

# Week 1 – Bayesian decoder
STIMULI = ['Left', 'Forward', 'Right']
PRIORS = np.array([0.20, 0.50, 0.30])
SPIKE_COUNTS = np.arange(0, 21)

# Week 1 – Tuning curve
TRUE_A = 50.0
TRUE_MU = 90.0
TRUE_SIG = 25.0
TRUE_B = 5.0
N_REPS = 10
NOISE_STD_TUNING = 6.0
ORIENTATIONS = np.arange(0, 181, 15)

# Week 2 – Population decoding
N_NEURONS = 8
NOISE_STD_POP = 0.05
TRUE_ANGLES_DEG = np.arange(0, 360, 15)