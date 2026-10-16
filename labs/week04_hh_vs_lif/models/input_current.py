# models/input_current.py
import numpy as np

def step_current(t, t_on=10.0, t_off=60.0, amp=10.0):
    """Step current: amp during [t_on, t_off], else 0. (t in ms)"""
    I = np.zeros_like(t)
    I[(t >= t_on) & (t <= t_off)] = amp
    return I

def pulse_train(t, pulses):
    """
    pulses: list of tuples [(t_on, t_off, amp), ...]
    """
    I = np.zeros_like(t)
    for (on, off, amp) in pulses:
        I[(t >= on) & (t <= off)] += amp
    return I

def noisy_current(t, base=10.0, noise_std=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_std, size=len(t))