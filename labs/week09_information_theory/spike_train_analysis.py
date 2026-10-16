import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Spike Train Binning
# -----------------------------------------

def bin_spike_train(spike_times, duration, bin_size):
    """
    Convert spike times to binned spike count vector.
    """

    # TODO 1:
    # Create bins from 0 to duration
    n_bins = int(np.ceil(duration / bin_size))
    edges = np.arange(n_bins + 1) * bin_size

    # TODO 2:
    # Count spikes in each bin
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts


def compute_spike_train_entropy(spike_trains, duration, bin_size):
    """
    Compute entropy of spike patterns across trials.
    """

    # TODO 3:
    # Convert each spike train into binned representation
    patterns = [tuple(bin_spike_train(train, duration, bin_size))
                for train in spike_trains]

    # TODO 4:
    # Treat each unique pattern as symbol
    unique_patterns = sorted(set(patterns))
    counts = np.array([patterns.count(u) for u in unique_patterns])

    # Compute entropy
    probabilities = counts / counts.sum()
    return compute_entropy(probabilities)