# =============================================================================
# IT5092 Assignment — utils/metrics.py
# Classification Metrics — PROVIDED, no TODOs
# =============================================================================

import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) — ground truth labels
    y_pred : np.ndarray, shape (N,) — predicted labels

    Returns
    -------
    accuracy : float in [0, 1]
    """
    return np.mean(y_true == y_pred)


def assign_neuron_labels(spike_counts: np.ndarray,
                         labels: np.ndarray,
                         n_classes: int = 10) -> np.ndarray:
    """
    Assign each excitatory neuron to the digit class that caused it to fire most.

    This is the unsupervised label assignment strategy from Diehl & Cook (2015).
    After training, we look back at the training data and ask: for each neuron,
    which digit class made it fire the most on average? That digit becomes the
    neuron's label.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_images, n_neurons)
        Spike count for each neuron on each training image.
    labels : np.ndarray, shape (n_images,)
        True digit label for each training image (integers 0–9).
    n_classes : int
        Number of classes (10 for MNIST).

    Returns
    -------
    assignments : np.ndarray, shape (n_neurons,)
        The digit class (0–9) assigned to each excitatory neuron.
    """
    n_neurons = spike_counts.shape[1]
    assignments = np.zeros(n_neurons, dtype=int)

    for j in range(n_neurons):
        mean_counts = np.zeros(n_classes)
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                mean_counts[c] = np.mean(spike_counts[idx, j])
        assignments[j] = np.argmax(mean_counts)

    return assignments


def predict_from_spikes(spike_counts: np.ndarray,
                        assignments: np.ndarray,
                        n_classes: int = 10) -> np.ndarray:
    """
    Classify each image by looking at which class has the most active neurons.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_images, n_neurons)
        Spike counts collected during the test phase.
    assignments : np.ndarray, shape (n_neurons,)
        Neuron-to-class assignments from assign_neuron_labels().
    n_classes : int
        Number of classes.

    Returns
    -------
    predictions : np.ndarray, shape (n_images,)
        Predicted digit class for each test image.
    """
    n_images = spike_counts.shape[0]
    predictions = np.zeros(n_images, dtype=int)

    for i in range(n_images):
        class_activity = np.zeros(n_classes)
        for c in range(n_classes):
            neuron_idx = np.where(assignments == c)[0]
            if len(neuron_idx) > 0:
                class_activity[c] = np.sum(spike_counts[i, neuron_idx])
        predictions[i] = np.argmax(class_activity)

    return predictions


def classification_report_snn(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: list = None) -> str:
    """
    Print a simple per-class accuracy report.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N,)
    class_names : list of str, optional

    Returns
    -------
    report : str
    """
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    lines = [f"{'Class':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"]
    lines.append('-' * 40)
    for c, name in enumerate(class_names):
        mask    = y_true == c
        correct = np.sum(y_pred[mask] == c)
        total   = np.sum(mask)
        acc     = correct / total if total > 0 else 0.0
        lines.append(f"{name:<10} {correct:>8} {total:>8} {acc:>10.1%}")
    lines.append('-' * 40)
    overall = compute_accuracy(y_true, y_pred)
    lines.append(f"{'Overall':<10} {' ':>8} {len(y_true):>8} {overall:>10.1%}")
    report = '\n'.join(lines)
    print(report)
    return report
