"""
Supporting experiments used to answer the written questions in Lab Sheet 7.
(Not part of the TODOs - run separately: python experiments_for_questions.py)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from config import (SEED, STIMULI, PRIORS, SPIKE_COUNTS, TRUE_A, TRUE_MU,
                    TRUE_SIG, TRUE_B, N_REPS, ORIENTATIONS, TRUE_ANGLES_DEG)
from utils import r2_score, angular_error_deg, wrap_angle_deg


# ------------------------------------------------------------------
# Ex 1 - effect of priors (actual priors vs. uniform priors)
# ------------------------------------------------------------------
def gaussian_likelihood(counts, mu, sigma, scale=0.15):
    return scale * np.exp(-0.5 * ((counts - mu) / sigma) ** 2)

R_fine = np.linspace(0, 20, 20001)
lik = np.column_stack([gaussian_likelihood(R_fine, m, 3) for m in (5, 10, 15)])

def boundaries(priors):
    post = lik * priors
    post /= post.sum(axis=1, keepdims=True)
    dec = np.argmax(post, axis=1)
    return [round(R_fine[i], 2) for i in range(1, len(dec)) if dec[i] != dec[i - 1]]

print("=== Ex1: priors ===")
print("Continuous boundaries, priors [0.2,0.5,0.3]:", boundaries(PRIORS))
print("Continuous boundaries, uniform priors      :", boundaries(np.ones(3) / 3))
lik10 = np.array([gaussian_likelihood(10, m, 3) for m in (5, 10, 15)])
p10 = lik10 * PRIORS / np.sum(lik10 * PRIORS)
print("Posterior at R=10 (actual priors):", np.round(p10, 3))
p10u = lik10 / lik10.sum()
print("Posterior at R=10 (uniform)      :", np.round(p10u, 3))


# ------------------------------------------------------------------
# Ex 2 - tuning curve fit at different noise levels
# ------------------------------------------------------------------
def gaussian_tuning(theta, A, mu, sigma, b):
    return A * np.exp(-0.5 * ((theta - mu) / sigma) ** 2) + b

def fit_tuning(noise_std, seed=SEED):
    rng = np.random.default_rng(seed)
    all_rates = np.zeros((len(ORIENTATIONS), N_REPS))
    for idx, theta in enumerate(ORIENTATIONS):
        true_rate = gaussian_tuning(theta, TRUE_A, TRUE_MU, TRUE_SIG, TRUE_B)
        all_rates[idx] = np.clip(true_rate + rng.normal(0, noise_std, N_REPS), 0, None)
    mean_rates = all_rates.mean(axis=1)
    popt, _ = curve_fit(gaussian_tuning, ORIENTATIONS, mean_rates, p0=[40, 80, 20, 2], maxfev=10000)
    r2 = r2_score(mean_rates, gaussian_tuning(ORIENTATIONS, *popt))
    return popt, r2, all_rates

print("\n=== Ex2: noise_std comparison (seed 42) ===")
results = {}
for ns in (6.0, 20.0):
    popt, r2, rates = fit_tuning(ns)
    results[ns] = (popt, r2, rates)
    print(f"noise={ns:>4} Hz  A={popt[0]:.2f} mu={popt[1]:.2f} sigma={popt[2]:.2f} "
          f"b={popt[3]:.2f}  R2={r2:.4f}")

# average over many seeds for a fairer comparison
print("Average over 200 seeds:")
for ns in (6.0, 20.0):
    P, R = [], []
    for s in range(200):
        popt, r2, _ = fit_tuning(ns, seed=s)
        P.append(popt); R.append(r2)
    P = np.array(P); true = np.array([TRUE_A, TRUE_MU, TRUE_SIG, TRUE_B])
    print(f"noise={ns:>4} Hz  mean|err| A={np.mean(abs(P[:,0]-true[0])):.2f} "
          f"mu={np.mean(abs(P[:,1]-true[1])):.2f} sigma={np.mean(abs(P[:,2]-true[2])):.2f} "
          f"b={np.mean(abs(P[:,3]-true[3])):.2f}  mean R2={np.mean(R):.4f}")

theta_fine = np.linspace(0, 180, 300)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, ns in zip(axes, (6.0, 20.0)):
    popt, r2, rates = results[ns]
    for i, th in enumerate(ORIENTATIONS):
        ax.scatter([th] * N_REPS, rates[i], alpha=0.4, s=15, color='tab:blue')
    ax.errorbar(ORIENTATIONS, rates.mean(1), yerr=rates.std(1), fmt='o', color='k', capsize=3)
    ax.plot(theta_fine, gaussian_tuning(theta_fine, *popt), 'r-', lw=2, label='Fit')
    ax.plot(theta_fine, gaussian_tuning(theta_fine, TRUE_A, TRUE_MU, TRUE_SIG, TRUE_B),
            'g--', lw=1.5, label='True')
    ax.set_title(f"noise_std = {ns:.0f} Hz   (R² = {r2:.3f})")
    ax.set_xlabel("Orientation (°)"); ax.grid(True); ax.legend()
axes[0].set_ylabel("Firing rate (Hz)")
plt.tight_layout(); plt.savefig("q_ex2_noise_comparison.png", dpi=150); plt.close()


# ------------------------------------------------------------------
# Ex 3 - population vector: neuron count and noise sweeps
# ------------------------------------------------------------------
def pv_decode_mae(n_neurons, noise_std, n_trials=200, seed=SEED):
    rng = np.random.default_rng(seed)
    pref = np.deg2rad(np.linspace(0, 360, n_neurons, endpoint=False))
    vecs = np.column_stack([np.cos(pref), np.sin(pref)])
    true_rad = np.deg2rad(TRUE_ANGLES_DEG)
    errs = []
    for _ in range(n_trials):
        for th, th_deg in zip(true_rad, TRUE_ANGLES_DEG):
            r = np.maximum(0, np.cos(th - pref)) + rng.normal(0, noise_std, n_neurons)
            r = np.clip(r, 0, None)
            pv = r @ vecs
            dec = wrap_angle_deg(np.rad2deg(np.arctan2(pv[1], pv[0])))
            errs.append(angular_error_deg(dec, th_deg))
    return np.mean(np.abs(errs))

neuron_counts = [4, 6, 8, 16, 32, 64, 128]
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
print("\n=== Ex3: neuron count sweep (noise=0.05, MAE deg, 200 trials) ===")
mae_n = [pv_decode_mae(n, 0.05) for n in neuron_counts]
for n, m in zip(neuron_counts, mae_n):
    print(f"N={n:>4}: MAE={m:.2f}")
print("Noiseless N=4 MAE:", round(pv_decode_mae(4, 0.0, n_trials=1), 3),
      " N=3 MAE:", round(pv_decode_mae(3, 0.0, n_trials=1), 3))
print("\n=== Ex3: noise sweep (N=8, MAE deg, 200 trials) ===")
mae_s = [pv_decode_mae(8, s) for s in noise_levels]
for s, m in zip(noise_levels, mae_s):
    print(f"noise={s:.2f}: MAE={m:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(neuron_counts, mae_n, 'o-'); axes[0].set_xscale('log', base=2)
axes[0].set_xlabel("Number of neurons"); axes[0].set_ylabel("Mean absolute error (°)")
axes[0].set_title("Effect of neuron count (noise = 0.05)"); axes[0].grid(True)
axes[1].plot(noise_levels, mae_s, 'o-', color='tab:red')
axes[1].set_xlabel("Noise SD"); axes[1].set_ylabel("Mean absolute error (°)")
axes[1].set_title("Effect of noise (N = 8 neurons)"); axes[1].grid(True)
plt.tight_layout(); plt.savefig("q_ex3_sweeps.png", dpi=150); plt.close()

# non-uniform preferred directions (for reflection Q2)
print("\n=== Non-uniform preferred directions (8 neurons clustered in 0-90°, no noise) ===")
pref = np.deg2rad(np.linspace(0, 90, 8))
vecs = np.column_stack([np.cos(pref), np.sin(pref)])
errs = []
for th_deg in TRUE_ANGLES_DEG:
    r = np.maximum(0, np.cos(np.deg2rad(th_deg) - pref))
    pv = r @ vecs
    if np.allclose(pv, 0):
        errs.append(np.nan); continue
    dec = wrap_angle_deg(np.rad2deg(np.arctan2(pv[1], pv[0])))
    errs.append(angular_error_deg(dec, th_deg))
errs = np.array(errs)
print("MAE (decodable dirs):", round(np.nanmean(np.abs(errs)), 2),
      " undecodable (all silent):", int(np.isnan(errs).sum()), "of", len(errs))
