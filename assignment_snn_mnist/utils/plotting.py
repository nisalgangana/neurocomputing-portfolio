# =============================================================================
# IT5092 Assignment — utils/plotting.py
# Visualisation Utilities — PROVIDED, no TODOs
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_spike_raster(spike_times: dict, spike_neurons: dict,
                      duration_ms: float, title: str = 'Spike Raster',
                      figsize=(12, 4)):
    """
    Plot a spike raster for the excitatory population.

    Parameters
    ----------
    spike_times   : dict {label: array of spike times in ms}
                    e.g., {'Excitatory': spike_monitor.t / b2.ms}
    spike_neurons : dict {label: array of neuron indices}
                    e.g., {'Excitatory': spike_monitor.i}
    duration_ms   : float  simulation duration for x-axis scaling
    title         : str
    """
    fig, ax = plt.subplots(figsize=figsize)
    colours = ['steelblue', 'tomato', 'seagreen']
    for idx, (label, times) in enumerate(spike_times.items()):
        neurons = spike_neurons[label]
        ax.scatter(times, neurons, s=1, c=colours[idx % len(colours)],
                   label=label, alpha=0.6)
    ax.set_xlim(0, duration_ms)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(title)
    ax.legend(markerscale=6, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_weight_receptive_fields(weights: np.ndarray,
                                 n_input: int = 784,
                                 n_show: int = 25,
                                 title: str = 'Learned Receptive Fields',
                                 figsize=(12, 5)):
    """
    Display learned input→excitatory weights as 28×28 receptive field images.

    Parameters
    ----------
    weights  : np.ndarray, shape (n_input, n_exc)  — weight matrix
    n_input  : int  number of input neurons (784 for MNIST)
    n_show   : int  how many neurons to display (must be a perfect square)
    title    : str
    """
    grid = int(np.sqrt(n_show))
    assert grid * grid == n_show, "n_show must be a perfect square (e.g. 25)"
    img_size = int(np.sqrt(n_input))

    fig, axes = plt.subplots(grid, grid, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[1]:
            rf = weights[:, i].reshape(img_size, img_size)
            im = ax.imshow(rf, cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
    plt.suptitle(title, y=1.01)
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Weight')
    plt.tight_layout()
    plt.show()


def plot_firing_rate_distribution(spike_counts: np.ndarray,
                                  duration_ms: float,
                                  title: str = 'Firing Rate Distribution',
                                  figsize=(8, 4)):
    """
    Plot histogram of mean firing rates across excitatory neurons.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_neurons,) — total spike count per neuron
    duration_ms  : float  — total simulation time
    title        : str
    """
    rates_hz = spike_counts / (duration_ms * 1e-3)  # convert to Hz
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(rates_hz, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(rates_hz), color='tomato', linewidth=2,
               label=f'Mean: {np.mean(rates_hz):.1f} Hz')
    ax.set_xlabel('Mean firing rate (Hz)')
    ax.set_ylabel('Number of neurons')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(f"Firing rates — Mean: {np.mean(rates_hz):.2f} Hz, "
          f"Max: {np.max(rates_hz):.2f} Hz, "
          f"Silent neurons: {np.sum(spike_counts == 0)}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          title: str = 'Confusion Matrix',
                          figsize=(6, 5)):
    """
    Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    cm          : np.ndarray, shape (n_classes, n_classes)
                  Use sklearn.metrics.confusion_matrix to compute.
    class_names : list of str
    title       : str
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=9)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_training_progress(accuracies: list, interval: int,
                           title: str = 'Training Accuracy',
                           figsize=(8, 4)):
    """
    Line plot of classification accuracy over training.

    Parameters
    ----------
    accuracies : list of float  — accuracy at each log interval
    interval   : int  — images between each logged accuracy
    title      : str
    """
    x = [(i + 1) * interval for i in range(len(accuracies))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, accuracies, marker='o', markersize=4, linewidth=1.5,
            color='steelblue')
    ax.set_xlabel('Training images seen')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.axhline(0.1, color='gray', linestyle='--', linewidth=1,
               label='Chance (10%)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_neuron_assignments(assignments: np.ndarray,
                            n_classes: int = 10,
                            title: str = 'Neuron Class Assignments',
                            figsize=(8, 4)):
    """
    Bar chart: how many excitatory neurons are assigned to each digit class.

    Parameters
    ----------
    assignments : np.ndarray, shape (n_neurons,)
    n_classes   : int
    """
    counts = [np.sum(assignments == c) for c in range(n_classes)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(n_classes), counts, color='steelblue', edgecolor='white')
    ax.axhline(len(assignments) / n_classes, color='tomato', linestyle='--',
               label='Equal split')
    ax.set_xlabel('Digit class')
    ax.set_ylabel('Number of neurons')
    ax.set_title(title)
    ax.set_xticks(range(n_classes))
    ax.legend()
    plt.tight_layout()
    plt.show()
