import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Mutual Information
# -----------------------------------------

def compute_joint_distribution(stimulus, response):
    """
    Compute empirical joint probability P(S, R)
    """

    # TODO 1:
    # Find unique stimulus and response values
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    s_values, s_idx = np.unique(stimulus, return_inverse=True)
    r_values, r_idx = np.unique(response, return_inverse=True)

    # Count co-occurrences
    joint_counts = np.zeros((len(s_values), len(r_values)))
    np.add.at(joint_counts, (s_idx, r_idx), 1)

    # Normalize to probabilities
    joint_prob = joint_counts / joint_counts.sum()
    return joint_prob, s_values, r_values


def compute_mutual_information(stimulus, response):
    """
    Compute I(S ; R)
    """

    # TODO 2:
    # Compute joint distribution
    p_sr, _, _ = compute_joint_distribution(stimulus, response)

    # TODO 3:
    # Compute marginal P(S) and P(R)
    p_s = p_sr.sum(axis=1)          # sum over responses
    p_r = p_sr.sum(axis=0)          # sum over stimuli

    # TODO 4:
    # Use formula:
    # I = sum P(s,r) * log2( P(s,r) / (P(s)P(r)) )
    I = 0.0
    for i in range(len(p_s)):
        for j in range(len(p_r)):
            if p_sr[i, j] > 0:      # 0 * log(0) is taken as 0
                I += p_sr[i, j] * np.log2(p_sr[i, j] / (p_s[i] * p_r[j]))

    # Equivalent check: I(S;R) = H(S) + H(R) - H(S,R)
    return float(I)