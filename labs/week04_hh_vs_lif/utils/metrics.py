# utils/metrics.py
import numpy as np

def spike_times_from_threshold(V, t, thresh=0.0):
    """
    Detect spikes by upward threshold crossing.
    For HH, thresh=0mV often works.
    """
    crossings = np.where((V[:-1] < thresh) & (V[1:] >= thresh))[0]
    return t[crossings + 1]

def firing_rate(spike_times, t_start, t_end):
    dur_s = (t_end - t_start) / 1000.0  # ms -> s
    if dur_s <= 0:
        return 0.0
    return len(spike_times) / dur_s

def isi(spike_times):
    """Inter-spike intervals in ms."""
    return np.diff(spike_times)

def cv(values):
    """Coefficient of variation."""
    values = np.asarray(values)
    if len(values) < 2:
        return np.nan
    return values.std() / values.mean()