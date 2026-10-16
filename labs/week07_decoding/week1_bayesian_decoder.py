import numpy as np
import matplotlib.pyplot as plt
from config import STIMULI, PRIORS, SPIKE_COUNTS

def gaussian_likelihood(counts, mu, sigma, scale=0.15):
    return scale * np.exp(-0.5 * ((counts - mu) / sigma) ** 2)

# ------------------------------------------------------------
# Provided likelihood curves
# ------------------------------------------------------------
lik_left = gaussian_likelihood(SPIKE_COUNTS, mu=5, sigma=3)
lik_forward = gaussian_likelihood(SPIKE_COUNTS, mu=10, sigma=3)
lik_right = gaussian_likelihood(SPIKE_COUNTS, mu=15, sigma=3)

likelihoods = np.column_stack([lik_left, lik_forward, lik_right])

# ------------------------------------------------------------
# TODO 1: Compute posterior probabilities
# Hint:
# joint = likelihoods[i] * PRIORS
# marginal = sum of joint values
# posterior = joint / marginal
# ------------------------------------------------------------
posteriors = np.zeros_like(likelihoods)

for i in range(len(SPIKE_COUNTS)):
    joint = likelihoods[i] * PRIORS          # P(R|S) * P(S)
    marginal = np.sum(joint)                 # P(R) = sum_S P(R|S) P(S)
    posteriors[i] = joint / marginal         # P(S|R)

# ------------------------------------------------------------
# TODO 2: Find MAP decision for each spike count
# Hint:
# - Use np.argmax on each row of posteriors
# - Convert the winning index to the corresponding stimulus label
# ------------------------------------------------------------
map_decisions = [STIMULI[idx] for idx in np.argmax(posteriors, axis=1)]

# ------------------------------------------------------------
# TODO 3: Print formatted results table
# Show only even spike counts
# Hint:
# You can access posterior values using posteriors[i, 0], posteriors[i, 1], posteriors[i, 2]
# ------------------------------------------------------------
print(f"{'R':>3} {'P(Left|R)':>11} {'P(Fwd|R)':>10} {'P(Right|R)':>11} {'MAP':>8}")
print("-" * 60)

for i, R in enumerate(SPIKE_COUNTS):
    if R % 2 == 0:
        print(f"{R:>3} {posteriors[i, 0]:>11.3f} {posteriors[i, 1]:>10.3f} "
              f"{posteriors[i, 2]:>11.3f} {map_decisions[i]:>8}")

# ------------------------------------------------------------
# TODO 4: Plot likelihoods and posterior probabilities
# Create two vertically stacked subplots
#
# Top plot:
# - lik_left, lik_forward, lik_right against SPIKE_COUNTS
#
# Bottom plot:
# - posterior curves for Left, Forward, Right
# - add vertical lines where MAP decision changes
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# Top panel: likelihoods
axes[0].plot(SPIKE_COUNTS, lik_left, label='Left')
axes[0].plot(SPIKE_COUNTS, lik_forward, label='Forward')
axes[0].plot(SPIKE_COUNTS, lik_right, label='Right')
axes[0].set_ylabel("Likelihood P(R|S)")
axes[0].set_title("Likelihood Functions")
axes[0].legend()
axes[0].grid(True)

# Bottom panel: posteriors
for k, stim in enumerate(STIMULI):
    axes[1].plot(SPIKE_COUNTS, posteriors[:, k], label=stim)

# add decision boundary lines at the first spike count where the MAP decision changes
for i in range(1, len(SPIKE_COUNTS)):
    if map_decisions[i] != map_decisions[i - 1]:
        axes[1].axvline(SPIKE_COUNTS[i], color='gray', linestyle=':', linewidth=1.5)
        print(f"Decision boundary at R = {SPIKE_COUNTS[i]}: "
              f"{map_decisions[i - 1]} -> {map_decisions[i]}")

axes[1].set_xlabel("Spike count R")
axes[1].set_ylabel("Posterior P(S|R)")
axes[1].set_title("Posterior Probabilities and MAP Boundaries")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("ex1_bayesian_decoder.png", dpi=150)
plt.show()