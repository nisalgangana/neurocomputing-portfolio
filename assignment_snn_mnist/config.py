# =============================================================================
# IT5092 — Neuroscience & Neurocomputing
# Assignment: Spiking Neural Networks for MNIST Classification
# Configuration File — All hyperparameters live here.
# DO NOT hard-code values in the notebook. Import from this file instead.
# =============================================================================

import numpy as np

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Brian2 Simulation Timestep ---
DT = 0.5  # ms

# =============================================================================
# MNIST SNN Configuration
# Inspired by: Diehl & Cook (2015), "Unsupervised learning of digit recognition
# using spike-timing-dependent plasticity", Frontiers in Computational Neuroscience
# =============================================================================

# --- Network Architecture ---
N_INPUT      = 784   # 28x28 MNIST pixels, flattened
N_EXCITATORY = 400   # excitatory LIF neurons (main learning population)
N_INHIBITORY = 400   # inhibitory LIF neurons (one per excitatory, lateral inhibition)

# --- Excitatory LIF Neuron Parameters ---
V_REST_E   = -65.0   # mV, resting membrane potential
V_RESET_E  = -65.0   # mV, membrane potential after spike reset
V_THRESH_E = -52.0   # mV, base spike threshold (before adaptive component)
TAU_M_E    = 100.0   # ms, membrane time constant (slow, biologically realistic)
TAU_GE     = 1.0     # ms, excitatory synaptic conductance decay
TAU_GI     = 2.0     # ms, inhibitory synaptic conductance decay
E_EXC      = 0.0     # mV, excitatory reversal potential
E_INH_E      = -100.0  # mV, inhibitory reversal potential
REFRAC_E   = 5.0     # ms, absolute refractory period
E_INH_I = -85.0

# --- Inhibitory LIF Neuron Parameters ---
V_REST_I   = -60.0   # mV
V_RESET_I  = -45.0   # mV  (note: higher than excitatory reset)
V_THRESH_I = -40.0   # mV  (easier to fire than excitatory neurons)
TAU_M_I    = 10.0    # ms  (10x faster than excitatory — quick suppression)
REFRAC_I   = 2.0     # ms

# --- Threshold Adaptation / Intrinsic Plasticity ---
# Each spike increases theta; theta decays very slowly → homeostatic regulation
THETA_INIT   = 20.0   # mV, initial adaptive threshold offset
THETA_PLUS   = 0.05   # mV, threshold increment per spike
TAU_THETA    = 1e7    # ms, decay time constant (~2.8 hours — effectively permanent
                      #     on the timescale of one training session)
THETA_OFFSET = 20.0   # mV, fixed offset subtracted from theta in threshold condition

# --- STDP Parameters (Input → Excitatory plastic synapses) ---
# Asymmetric rule: LTP is 100x stronger than LTD
LTP_RATE      = 0.01    # A_plus:  weight increase rate (causal: pre fires before post)
LTD_RATE      = 0.0001  # A_minus: weight decrease rate (anti-causal: post before pre)
TAU_STDP_PRE  = 20.0    # ms, pre-synaptic trace (x_pre) decay time constant
# TAU_STDP_POST = 40.0    # ms, post-synaptic trace (x_post) decay time constant
TAU_POST1    = 20
TAU_POST2    = 40.0
W_MIN         = 0.0     # minimum synaptic weight (hard clip)
W_MAX         = 1.0     # maximum synaptic weight (hard clip)
W_INIT_MEAN   = 0.3     # mean of initial weight distribution
W_INIT_STD    = 0.1     # std of initial weight distribution

# --- Fixed (Non-Plastic) Connection Weights ---
W_EXC_TO_INH  = 10.4   # excitatory → inhibitory (strong, fast activation)
W_INH_TO_EXC  = 17.0   # inhibitory → excitatory (strong lateral suppression)

# --- MNIST Input Rate Encoding ---
# Pixel intensity (0–255) is converted to Poisson firing rate (Hz)
# rate = (pixel / 8.0) * INPUT_INTENSITY_BASE
INPUT_INTENSITY_BASE  = 2.0    # scaling factor (resulting max rate ≈ 63.75 Hz)
PRESENTATION_TIME     = 350.0  # ms, duration each image is presented to the network
REST_TIME             = 150.0  # ms, silent period between images (STDP consolidation)
MIN_SPIKES_THRESHOLD  = 5      # if total spikes < this, boost input and retry

# --- Training / Testing ---
N_TRAIN            = 1000   # number of training images (full MNIST = 60000, reduced for speed)
N_TEST             = 50   # number of test images
BATCH_LOG_INTERVAL = 50   # print progress every N images
