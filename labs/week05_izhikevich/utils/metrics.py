# utils/metrics.py
import numpy as np

def firing_rate(spike_times, t_start, t_end):
    duration_s = (t_end - t_start) / 1000.0
    if duration_s <= 0:
        return 0
    return len(spike_times) / duration_s

def isi(spike_times):
    return np.diff(spike_times)

def spike_count(spike_times):
    return len(spike_times)