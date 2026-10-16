import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Coding Efficiency
# -----------------------------------------

def compare_distributions():
    """
    Compare entropy of:
    - Uniform distribution
    - Skewed distribution
    """

    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    skewed = np.array([0.7, 0.1, 0.1, 0.1])

    # TODO:
    # Compute entropy of both
    H_uniform = compute_entropy(uniform)
    H_skewed = compute_entropy(skewed)
    H_max = np.log2(len(uniform))          # maximum possible entropy for 4 symbols

    # Print comparison
    print(f"{'Distribution':<14}{'Probabilities':<26}{'Entropy (bits)':>15}{'Efficiency':>12}")
    print("-" * 67)
    print(f"{'Uniform':<14}{str(uniform.tolist()):<26}{H_uniform:>15.4f}{H_uniform / H_max:>11.1%}")
    print(f"{'Skewed':<14}{str(skewed.tolist()):<26}{H_skewed:>15.4f}{H_skewed / H_max:>11.1%}")
    print(f"\nMaximum entropy for 4 states = log2(4) = {H_max:.1f} bits")
    more = "Uniform" if H_uniform > H_skewed else "Skewed"
    print(f"More efficient distribution: {more} "
          f"(+{abs(H_uniform - H_skewed):.4f} bits per symbol)")

    return H_uniform, H_skewed