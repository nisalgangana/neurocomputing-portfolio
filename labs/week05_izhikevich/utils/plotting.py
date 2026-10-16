# utils/plotting.py
import matplotlib.pyplot as plt

def plot_trace(t, y, title, ylabel):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_multiple_traces(t, traces, labels, title, ylabel="Membrane potential (mV)"):
    plt.figure()
    for y, lbl in zip(traces, labels):
        plt.plot(t, y, label=lbl)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()