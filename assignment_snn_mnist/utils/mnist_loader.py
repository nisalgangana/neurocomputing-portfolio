# =============================================================================
# IT5092 Assignment — utils/mnist_loader.py
# MNIST Data Loader — PROVIDED, no TODOs
# =============================================================================

import numpy as np


def load_mnist(n_train: int = 10000, n_test: int = 1000,
               random_seed: int = 42) -> tuple:
    """
    Load and return MNIST digit images and labels.

    Downloads automatically via sklearn on first call (~11 MB).

    Parameters
    ----------
    n_train : int
        Number of training samples to return (max 60000).
    n_test : int
        Number of test samples to return (max 10000).
    random_seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, 784)
        Training images, pixel values in [0, 255].
    y_train : np.ndarray, shape (n_train,)
        Training labels, integers 0-9.
    X_test : np.ndarray, shape (n_test, 784)
        Test images, pixel values in [0, 255].
    y_test : np.ndarray, shape (n_test,)
        Test labels, integers 0-9.
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST via sklearn (downloads on first use)...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                             parser='auto')
        X = mnist.data.astype(np.float32)    # shape (70000, 784), values 0-255
        y = mnist.target.astype(np.int32)    # shape (70000,), labels 0-9

        # Split: MNIST is pre-split at 60000/10000
        X_train_full, X_test_full = X[:60000], X[60000:]
        y_train_full, y_test_full = y[:60000], y[60000:]

        # Subsample if requested
        rng = np.random.default_rng(random_seed)
        train_idx = rng.permutation(60000)[:n_train]
        test_idx  = rng.permutation(10000)[:n_test]

        print(f"MNIST loaded: {n_train} training, {n_test} test samples.")
        return (X_train_full[train_idx], y_train_full[train_idx],
                X_test_full[test_idx],  y_test_full[test_idx])

    except Exception as e:
        raise RuntimeError(
            f"Failed to load MNIST: {e}\n"
            "Install sklearn: pip install scikit-learn"
        )


def show_sample_images(X: np.ndarray, y: np.ndarray,
                       n_per_class: int = 2, figsize=(12, 3)):
    """
    Display sample MNIST images for each digit class (0-9).

    Parameters
    ----------
    X : np.ndarray, shape (N, 784)
    y : np.ndarray, shape (N,)
    n_per_class : int
        How many examples to show per digit.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_per_class, 10, figsize=figsize)
    for digit in range(10):
        idx = np.where(y == digit)[0][:n_per_class]
        for row, i in enumerate(idx):
            ax = axes[row, digit] if n_per_class > 1 else axes[digit]
            ax.imshow(X[i].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if row == 0:
                ax.set_title(str(digit), fontsize=10)
    plt.suptitle('Sample MNIST Images (one column per digit)', y=1.02)
    plt.tight_layout()
    plt.show()
