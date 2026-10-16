# utils/plotting.py
import matplotlib.pyplot as plt

def plot_trace(t, y, title, ylabel):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.show()

def plot_hh_gates(t, m, h, n):
    plt.figure()
    plt.plot(t, m, label="m")
    plt.plot(t, h, label="h")
    plt.plot(t, n, label="n")
    plt.title("HH gating variables")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend()
    plt.show()