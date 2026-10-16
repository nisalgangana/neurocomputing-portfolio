import numpy as np

from config import EPOCHS, LEARNING_RATE, SEED

def compute_output(x, w):
    return np.dot(w, x)

def hebbian_update(w, x, y, lr):
    """
    Hebbian learning rule:
    w_new = w + lr * x * y
    """
    # TODO 1:
    # Implement the Hebbian update rule
    # Each weight grows in proportion to the product of its input x_i and
    # the output y ("cells that fire together, wire together").
    w_new = w + lr * x * y
    return w_new

def train_hebbian(X, epochs=EPOCHS, lr=LEARNING_RATE, w_init=None):
    n_features = X.shape[1]

    # NOTE: If w starts at exactly zero, then y = w . x = 0 for every input,
    # so the update lr * x * y is always 0 and the weights never change.
    # A small random initial weight vector is used so that learning can start.
    if w_init is None:
        rng = np.random.default_rng(SEED)
        w = rng.uniform(0.0, 0.1, size=n_features)
    else:
        w = np.array(w_init, dtype=float)

    history = []

    for epoch in range(epochs):
        for x in X:
            y = compute_output(x, w)
            w = hebbian_update(w, x, y, lr)
        history.append(w.copy())

    return w, np.array(history)


# ---------------------------------------------------------------------------
# Optional challenge: normalized Hebbian learning (Oja's rule)
# ---------------------------------------------------------------------------
def oja_update(w, x, y, lr):
    """
    Oja's rule:
    w_new = w + lr * y * (x - y * w)
    The extra -lr * y^2 * w term acts as weight decay that keeps ||w|| close to 1.
    """
    return w + lr * y * (x - y * w)

def train_oja(X, epochs=EPOCHS, lr=LEARNING_RATE, w_init=None):
    n_features = X.shape[1]
    if w_init is None:
        rng = np.random.default_rng(SEED)
        w = rng.uniform(0.0, 0.1, size=n_features)
    else:
        w = np.array(w_init, dtype=float)

    history = []

    for epoch in range(epochs):
        for x in X:
            y = compute_output(x, w)
            w = oja_update(w, x, y, lr)
        history.append(w.copy())

    return w, np.array(history)
