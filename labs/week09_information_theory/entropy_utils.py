import numpy as np

# -----------------------------------------
# Entropy Computation
# -----------------------------------------

def compute_entropy(probabilities):
    """
    Compute Shannon entropy H(X) in bits.

    Parameters:
        probabilities (array-like): probability distribution (must sum to 1)

    Returns:
        float: entropy in bits
    """

    # TODO 1:
    # Remove zero probabilities to avoid log(0)
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]

    # TODO 2:
    # Compute entropy using:
    # H = -sum(p * log2(p))
    H = -np.sum(p * np.log2(p))
    return float(H) + 0.0   # "+ 0.0" turns -0.0 into 0.0


def normalize_distribution(counts):
    """
    Convert raw counts into probability distribution.
    """

    # TODO 3:
    # Normalize counts so they sum to 1
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        raise ValueError("Counts sum to zero - cannot normalize.")
    return counts / total