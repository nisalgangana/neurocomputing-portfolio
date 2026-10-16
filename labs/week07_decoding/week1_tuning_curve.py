import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from config import (
    rng, TRUE_A, TRUE_MU, TRUE_SIG, TRUE_B,
    N_REPS, NOISE_STD_TUNING, ORIENTATIONS
)
from utils import r2_score

# ------------------------------------------------------------
# TODO 1: Define Gaussian tuning function
# Hint:
# A * exp(-0.5 * ((theta - mu)/sigma)^2) + b
# ------------------------------------------------------------
def gaussian_tuning(theta, A, mu, sigma, b):
    return A * np.exp(-0.5 * ((theta - mu) / sigma) ** 2) + b

# ------------------------------------------------------------
# TODO 2: Simulate repeated noisy firing rates
# For each orientation:
# - compute the true rate using gaussian_tuning
# - add Gaussian noise using rng.normal
# - clip values so rates are not negative
# - store results in all_rates
# Then compute mean_rates and std_rates
# ------------------------------------------------------------
all_rates = np.zeros((len(ORIENTATIONS), N_REPS))

for idx, theta in enumerate(ORIENTATIONS):
    true_rate = gaussian_tuning(theta, TRUE_A, TRUE_MU, TRUE_SIG, TRUE_B)
    noisy_rates = true_rate + rng.normal(0, NOISE_STD_TUNING, N_REPS)
    noisy_rates = np.clip(noisy_rates, 0, None)   # firing rates can't be negative
    all_rates[idx] = noisy_rates

mean_rates = np.mean(all_rates, axis=1)
std_rates = np.std(all_rates, axis=1)

# ------------------------------------------------------------
# TODO 3: Fit Gaussian tuning curve to mean rates
# Hint:
# - Use curve_fit
# - Start with p0 = [40, 80, 20, 2]
# - Fit using ORIENTATIONS and mean_rates
# ------------------------------------------------------------
p0 = [40, 80, 20, 2]
popt, pcov = curve_fit(gaussian_tuning, ORIENTATIONS, mean_rates, p0=p0)

A_fit, mu_fit, sig_fit, b_fit = popt

print("--- Fitted parameters ---")
print(f"A     : {A_fit:.2f} (true {TRUE_A})")
print(f"mu    : {mu_fit:.2f} (true {TRUE_MU})")
print(f"sigma : {sig_fit:.2f} (true {TRUE_SIG})")
print(f"b     : {b_fit:.2f} (true {TRUE_B})")

# ------------------------------------------------------------
# TODO 4: Compute R²
# Hint:
# fitted_curve = gaussian_tuning(ORIENTATIONS, *popt)
# then call r2_score(mean_rates, fitted_curve)
# ------------------------------------------------------------
fitted_curve = gaussian_tuning(ORIENTATIONS, *popt)
R2 = r2_score(mean_rates, fitted_curve)
print(f"R² : {R2:.4f}")

# ------------------------------------------------------------
# TODO 5: Plot the tuning data and fitted Gaussian
# Include:
# - scatter of all individual measurements
# - error bars for mean ± std
# - smooth fitted Gaussian curve
# - vertical dashed line at mu_fit
# ------------------------------------------------------------
theta_fine = np.linspace(0, 180, 300)
fit_fine = gaussian_tuning(theta_fine, *popt)

fig, ax = plt.subplots(figsize=(9, 5))

# scatter individual measurements
for i, theta in enumerate(ORIENTATIONS):
    ax.scatter([theta] * N_REPS, all_rates[i], alpha=0.5, s=20,
               label='Individual' if i == 0 else None)

# error bars for mean ± std
ax.errorbar(ORIENTATIONS, mean_rates, yerr=std_rates, fmt='o', color='black',
            capsize=4, label='Mean ± SD')
# fitted Gaussian curve
ax.plot(theta_fine, fit_fine, 'r-', linewidth=2, label='Gaussian fit')
# vertical line at preferred orientation
ax.axvline(mu_fit, color='gray', linestyle='--', label=f'μ fit = {mu_fit:.1f}°')

ax.set_xlabel("Orientation (°)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("Gaussian Tuning Curve Fit")
ax.legend()
ax.grid(True)

# Optional annotation box
text_str = (
    f"A={A_fit:.2f}\n"
    f"μ={mu_fit:.2f}\n"
    f"σ={sig_fit:.2f}\n"
    f"b={b_fit:.2f}\n"
    f"R²={R2:.3f}"
)
ax.text(0.02, 0.98, text_str, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', alpha=0.2))

plt.tight_layout()
plt.savefig("ex2_tuning_curve.png", dpi=150)
plt.show()