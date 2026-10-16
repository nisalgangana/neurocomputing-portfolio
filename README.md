# IT5092 Neuroscience & Neurocomputing: Learning Portfolio

**Nisal Gangana Wickramathilaka** (MS26910222)
MSc in Artificial Intelligence, SLIIT

This repository holds my lab work, assignment and learning portfolio for IT5092.

## Start here

**[IT5092_Learning_Portfolio_MS26910222.md](IT5092_Learning_Portfolio_MS26910222.md)**: the portfolio itself, with core concepts, selected lab evidence, assignment insight, a debugging case study and comparisons. Figures are in `portfolio_figures/`.

## Repository layout

| Folder | Content |
|---|---|
| `labs/week01_eeg_rhythms/` | EEG rhythm analysis on the PhysioNet EEGBCI dataset (MNE): preprocessing, Welch PSD, band power, alpha vs theta segment comparison |
| `labs/week02_eeg_physics/` | Physics of EEG: neuronal synchrony, dipole depth (1/r²), orientation (cos θ) and two-dipole interference |
| `labs/week03_spike_trains/` | Poisson spike trains, refractory periods, rate vs timing codes, population decoding |
| `labs/week04_hh_vs_lif/` | Hodgkin-Huxley vs Leaky Integrate-and-Fire: F-I curve, gating variables, pulse-gap-pulse refractory experiment |
| `labs/week05_izhikevich/` | Izhikevich model: regular spiking, fast spiking and bursting neurons |
| `labs/week06_hebbian/` | Hebbian learning, Oja's rule and a Hopfield-style associative memory |
| `labs/week07_decoding/` | Bayesian decoder, Gaussian tuning-curve fitting and population-vector decoding |
| `labs/week09_information_theory/` | Entropy, mutual information, spike-train binning and coding efficiency |
| `assignment_snn_mnist/` | Spiking neural network for MNIST in Brian2 (Diehl & Cook, 2015): conductance-based LIF, triplet STDP, winner-take-all inhibition, adaptive threshold |

## Running the code

```bash
pip install -r requirements.txt
jupyter notebook
```

Notes:

- The Week 1 notebook downloads the PhysioNet EEGBCI dataset through MNE on first run (the raw EDF files are not stored in this repository). Update the `MNE_DATA` path in the notebook to a folder on your machine.
- The assignment notebook loads MNIST through scikit-learn and runs in about 12 minutes on the NumPy backend for 1,000 training images.
- Each lab folder has its own `config.py`; run the notebooks from inside their folder so the local imports resolve.

## Assignment summary

Unsupervised digit recognition with 784 Poisson inputs, 400 excitatory and 400 inhibitory LIF neurons. Trained on 1,000 images, tested on 50: **64.0% test accuracy** (chance = 10%). The notebook includes a note on switching the integrator from forward Euler to exponential Euler after the network became numerically unstable under strong lateral inhibition.

## Reference

Diehl, P.U. & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
