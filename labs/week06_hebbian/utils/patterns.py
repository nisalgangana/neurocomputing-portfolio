import numpy as np

from config import NOISE_LEVELS

def make_demo_patterns():
    p1 = np.array([
        1, 1, 1, 1,
        1,-1,-1, 1,
        1,-1,-1, 1,
        1, 1, 1, 1
    ])

    p2 = np.array([
        1,-1,-1, 1,
        -1, 1, 1,-1,
        -1, 1, 1,-1,
        1,-1,-1, 1
    ])

    p3 = np.array([
        1, 1,-1,-1,
        1, 1,-1,-1,
        -1,-1, 1, 1,
        -1,-1, 1, 1
    ])

    return np.stack([p1, p2, p3])

def add_noise(pattern, noise_level=NOISE_LEVELS, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    noisy = pattern.copy()
    n_flip = int(len(pattern) * noise_level)
    idx = rng.choice(len(pattern), size=n_flip, replace=False)
    noisy[idx] *= -1
    return noisy