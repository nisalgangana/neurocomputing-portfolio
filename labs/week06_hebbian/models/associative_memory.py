import numpy as np

def train_associative_memory(patterns):
    """
    Store bipolar patterns using outer-product Hebbian learning
    """
    n_features = patterns.shape[1]
    W = np.zeros((n_features, n_features))

    for p in patterns:
        # TODO 4:
        # Add outer product of p with itself
        # W_ij grows when units i and j have the same sign in a pattern
        # and shrinks when they have opposite signs.
        W += np.outer(p, p)

    np.fill_diagonal(W, 0)
    return W

def recall_pattern(W, cue, steps=5):
    x = cue.copy()

    for _ in range(steps):
        # TODO 5:
        # Update pattern using sign(W @ x)
        h = W @ x
        x_new = np.sign(h)
        # np.sign returns 0 when the net input is exactly 0; in that case the
        # unit keeps its previous state so the pattern stays bipolar (+1/-1).
        x_new[x_new == 0] = x[x_new == 0]
        x = x_new.astype(cue.dtype)

    return x
