# IT5092 · Neuroscience & Neurocomputing — Learning Portfolio

| | |
|---|---|
| **Name** | Nisal Gangana Wickramathilaka |
| **Student ID** | MS26910222 |
| **Programme** | MSc in Artificial Intelligence, SLIIT |
| **Code repository** | `[Insert GitHub repository link here]` |

> **How to read this portfolio.** I have kept it selective on purpose. Each section shows how my understanding developed, not just what files I produced. Coming from a computing background into neuroscience, I often found that an idea became concrete only when I could connect the biology to the equation, the equation to the code, and the code to a result I could check. That chain is how most of this portfolio is written.

---

## Contents

1. [Core Concepts](#1-core-concepts)
2. [Selected Lab Evidence](#2-selected-lab-evidence)
3. [Assignment Insight — SNN for MNIST](#3-assignment-insight--spiking-neural-network-for-mnist)
4. [Mistakes and Debugging](#4-mistakes-and-debugging)
5. [Comparison and Synthesis](#5-comparison-and-synthesis)
6. [Concepts I Am Still Consolidating](#6-concepts-i-am-still-consolidating)

---

## 1. Core Concepts

### 1.1 Brain and behaviour

The brain has about 86 billion neurons. They communicate electrically *within* a cell (action potentials travelling along the axon) and chemically *between* cells (neurotransmitters released at synapses). Behaviour emerges from how these signals are routed through specialised regions:

- **Cortex lobes:** frontal (planning, decisions, motor control), parietal (touch, spatial awareness), temporal (hearing, memory, language), occipital (vision).
- **Subcortical systems:** the cerebellum (coordination, motor learning), the brainstem (breathing, heart rate, sleep–wake) and the limbic system (amygdala for fear and emotion, hippocampus for memory formation).
- **Neurotransmitters** change the "mode" of a circuit: glutamate (excitation, learning), GABA (inhibition), dopamine (reward, motivation), serotonin (mood).

What stayed with me is **neuroplasticity**: the wiring is not fixed. Repeated activity strengthens pathways. That is the biological basis for every learning rule I later implemented (Hebbian learning, Oja's rule, STDP).

### 1.2 EEG: what it actually measures

EEG does **not** record single neurons. It records the **summed postsynaptic potentials of large populations of pyramidal neurons** whose dendrites are aligned in parallel, measured through the skull and scalp. Lab 2 made three physical facts concrete for me:

1. **Synchrony dominates amplitude.** In-phase sources add linearly (RMS ∝ N). Random-phase sources partly cancel (RMS ∝ √N).
2. **Depth matters.** Dipole potential falls off as $V \propto p / r^2$, so deep sources barely reach the scalp.
3. **Orientation matters.** $V \propto \cos\theta$. Radial dipoles (gyri) are visible, while tangential dipoles (θ ≈ 90°) are nearly invisible.

Brain rhythms are the frequency "vocabulary" of these population signals:

| Band | Range | Typical state |
|---|---|---|
| Delta | 1–4 Hz | Deep sleep |
| Theta | 4–8 Hz | Drowsiness, memory processes |
| Alpha | 8–12 Hz | Relaxed wakefulness, eyes closed (strongest occipitally) |
| Beta | 13–30 Hz | Alert, focused, motor activity |
| Gamma | 30–45+ Hz | Binding, high-level processing (easily contaminated by muscle artefacts) |

The standard pipeline I used was: **raw µV signal → re-reference → band-pass 1–45 Hz → 50 Hz notch (Sri Lankan mains) → Welch PSD → band-power features → interpretation.** From a software perspective this is an ETL pipeline. Every stage removes a known class of noise before features are extracted.

### 1.3 Neuroimaging basics

**fMRI and the BOLD signal.** fMRI measures neural activity *indirectly*. Active neurons trigger an oversupply of oxygenated blood (neurovascular coupling). Deoxyhaemoglobin is paramagnetic and distorts the local magnetic field, while oxyhaemoglobin is not. A rise in the oxy/deoxy ratio therefore increases the MR signal. This is the **Blood-Oxygen-Level-Dependent (BOLD)** response. It is slow: the haemodynamic response function (HRF) peaks around 5–6 s after the neural event. That is why fMRI has excellent spatial resolution but poor temporal resolution.

**The GLM: one regression per voxel.** The General Linear Model answers *where* task-related activity happens:

$$\mathbf{y}_v = \mathbf{X}\boldsymbol{\beta}_v + \boldsymbol{\varepsilon}_v$$

- $\mathbf{y}_v$ is the BOLD time series of a single voxel $v$ (T time points).
- $\mathbf{X}$ is the **design matrix**. Each column is a task regressor: the stimulus timing (e.g. a boxcar of "See" vs "Blind" trials) **convolved with the HRF**, plus nuisance regressors such as head-motion parameters and drift.
- $\boldsymbol{\beta}_v$ are the fitted weights, estimated by least squares: $\hat{\boldsymbol\beta}_v = (\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top \mathbf y_v$.
- A **contrast** such as $c^\top\hat\beta$ for "See − Blind" gives a t-statistic per voxel. The **threshold** step (p < 0.05, corrected for multiple comparisons) produces the statistical brain map.

The full pipeline from the lecture is: **Preprocess** (slice-timing correction, motion correction, spatial smoothing) → **Model** (design matrix) → **Fit** (every voxel) → **Threshold**.

> **A computational view.** The GLM is *embarrassingly parallel*. The same design matrix $\mathbf X$ is shared by all ~100,000 voxels, and each voxel's fit is independent. So the pseudo-inverse $\mathbf X^{+}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ can be computed **once**, and all voxels solved in a single matrix multiplication: $\hat{\mathbf B} = \mathbf X^{+}\mathbf Y$, where $\mathbf Y$ is a $T \times V$ matrix. In practice this means a vectorised NumPy/BLAS call, or the voxel set split across cores, GPUs or cluster nodes with no communication between workers. Only the final multiple-comparison correction needs a global view. I find this a clean example of how a statistical idea and an efficient implementation fit together.

**Structural MRI: the B0 field and hydrogen protons.** Structural MRI images anatomy, not activity. The body is full of hydrogen nuclei (protons in water and fat), and each behaves like a tiny spinning magnet. Normally their orientations are random, so the net magnetisation is zero. Inside the scanner's strong static field **B0** (e.g. 1.5 T or 3 T), a small net excess of protons aligns *with* the field. This creates a net magnetisation vector along B0, and the protons **precess** around B0 at the Larmor frequency $\omega_0 = \gamma B_0$ (≈ 42.58 MHz per tesla, so ≈ 128 MHz at 3 T). A radio-frequency pulse at exactly that frequency tips the magnetisation away from B0. As it relaxes back (T1 recovery, T2 decay), the protons emit a signal. Grey matter, white matter and CSF relax at different rates, and that difference is the source of image contrast. Gradient coils vary the field slightly across space so that each location has a slightly different frequency, which is how the signal is mapped back to positions.

**Other methods, briefly.** MEG (magnetic fields from neural currents, good timing), PET (radioactive tracers for metabolism and receptors, invasive), fNIRS (portable cortical haemodynamics), DTI (white-matter tracts). The lecture's key message, which I agree with: *no single method is best. The question decides the tool.*

### 1.4 Neural coding: encoding and decoding

- **Encoding** asks how a stimulus produces a response: $P(R \mid S)$. Examples are tuning curves and rate models.
- **Decoding** asks what a response tells us about the stimulus: $P(S \mid R)$ or $\hat S = f(R)$.
- **Bayes connects them:** $P(S\mid R) = \dfrac{P(R\mid S)\,P(S)}{P(R)}$. An encoding model plus a prior gives a decoder.

The forms of neural code I worked with:

- **Rate code:** information is in spike count per window. It is robust but slow, because it needs counting time.
- **Temporal code:** information is in precise spike timing. It is fast and high-capacity, but fragile to jitter.
- **Population code:** information is spread across many neurons. Independent noise averages out roughly as $1/\sqrt{N}$.

Spike trains are commonly modelled as **Poisson processes** with $P(\text{spike in } dt) = r\,dt$. A **refractory period** caps the maximum rate at roughly $1/t_{ref}$ and removes short inter-spike intervals.

### 1.5 Information theory

- **Entropy:** $H(R) = -\sum p \log_2 p$ measures how variable (uncertain) a response is, i.e. its *capacity*.
- **Mutual information:** $I(S;R) = H(R) - H(R\mid S)$ measures how much of that variability is actually *about* the stimulus.
- **Noise** increases $H(R\mid S)$ and therefore lowers MI. A neuron can have high entropy and near-zero MI if its variability is just noise.
- **Bin size** changes the measured code: large bins turn a timing code into a count code and discard information.
- **Efficient coding** (Barlow, Laughlin): using all response levels equally often maximises information per spike, which matters because spikes are metabolically expensive.

### 1.6 Neuron models

| Model | Core equation(s) | What it captures |
|---|---|---|
| **Hodgkin–Huxley** | $C_m \dot V = I - g_{Na}m^3h(V-E_{Na}) - g_K n^4 (V-E_K) - g_L(V-E_L)$, with $\dot x = \alpha_x(V)(1-x) - \beta_x(V)x$ for $x \in \{m,h,n\}$ | Real ionic mechanism: Na⁺ upstroke, K⁺ repolarisation, refractoriness from h-inactivation |
| **LIF** | $\tau_m \dot V = -(V-V_{rest}) + RI$; spike and reset at threshold | Integration plus threshold only. Cheap; refractoriness is a hand-coded rule |
| **Conductance-based LIF** (assignment) | $\tau_m \dot v = -(v - V_{rest}) + g_e(E_{exc}-v) + g_i(E_{inh}-v)$ | Synaptic input depends on driving force; voltage is bounded by reversal potentials |
| **Izhikevich** | $\dot v = 0.04v^2 + 5v + 140 - u + I$, $\dot u = a(bv-u)$; reset $v\leftarrow c,\ u \leftarrow u+d$ | Many cortical firing patterns (regular, fast, bursting) from 4 parameters at near-LIF cost |

### 1.7 Network-based computation and learning

- **Hebbian learning** ($\Delta w = \eta\,x\,y$, "fire together, wire together") finds the dominant correlation in the input, which is the first principal component. On its own it is **unstable** (exponential weight growth).
- **Oja's rule** adds a decay term $-\eta y^2 w$ that keeps $\lVert w\rVert \to 1$: Hebbian learning with built-in normalisation.
- **Associative (Hopfield-style) memory** stores patterns as $W = \sum_p \mathbf p\mathbf p^\top$ and recalls them by iterating $\mathbf x \leftarrow \operatorname{sign}(W\mathbf x)$. Stored patterns become **attractors** of a recurrent network.
- **STDP** makes Hebbian learning *causal*: pre-before-post gives LTP, post-before-pre gives LTD.
- **Winner-take-all** via lateral inhibition plus **homeostasis** (adaptive threshold θ) lets a population divide up the input space without labels (Diehl & Cook, 2015).

---

## 2. Selected Lab Evidence

I chose four outputs that each changed how I think about the subject. Other labs are summarised in a short table at the end of this section.

### Evidence 1 — Lab 2: why EEG measures *synchrony*, not *number of neurons*

**What I produced.** I simulated 50 neurons oscillating at 10 Hz, once in phase and once with random phases, then swept N from 5 to 200. I also summed two dipole fields at two separations.

| Metric (N = 50) | Synchronous | Asynchronous |
|---|---|---|
| RMS amplitude | **35.34** | **4.58** |
| Peak amplitude | 50.00 | 6.48 |

`[Insert Screenshot of "RMS vs Number of Neurons" from Week 2 here — portfolio_figures/W2_rms_vs_N.png]`

`[Insert Screenshot of two-dipole fields (far ±0.3 vs close ±0.1) from Week 2 here — portfolio_figures/W2_two_dipole_far.png and W2_two_dipole_close.png]`

**Reflection.** The same 50 neurons produced almost an **8× difference** in signal just by changing their phase alignment. Synchronous RMS grew linearly with N, asynchronous RMS only as √N. Before this lab I assumed a "bigger EEG signal" meant "more neurons active". It actually means "more neurons active *together*". The two-dipole plot showed the other side: at ±0.1 the two sources merged into one blob. EEG cannot tell them apart, which is the physical reason behind its poor spatial resolution. Depth gave the same message numerically: peak potential fell from 2500 to 25 as depth went from 0.02 to 0.20 ($1/r^2$).

---

### Evidence 2 — Lab 4: the refractory period *emerges* from Hodgkin–Huxley

**What I produced.** I implemented the HH gating updates and ionic currents, then ran a **pulse–gap–pulse** experiment: two 1 ms, 20 µA/cm² pulses with the gap swept from 1 to 20 ms.

| Gap (ms) | 1 … 10 | 11 … 20 |
|---|---|---|
| Second spike? | No | **Yes** |

Estimated HH refractory period: **≈ 11 ms**.

`[Insert Screenshot of HH pulse–gap refractory curve from Week 4 here — portfolio_figures/W4_hh_pulse_gap_refractory.png]`

`[Insert Screenshot of gap = 4 ms vs gap = 14 ms with m/h/n gates from Week 4 here — portfolio_figures/W4_hh_h_recovery_gap4_vs_14.png]`

**Reflection.** The gate plot was the moment HH "clicked" for me. At a 4 ms gap, the Na⁺ inactivation gate **h** has dropped to about 0.1 and has not recovered, so the second pulse cannot open enough Na⁺ channels. At 14 ms, h has climbed back to about 0.6 and a full second spike fires. Nobody wrote "refractory period = 11 ms" into the HH code; it *falls out of the channel kinetics*. In my LIF model the refractory period was a single line (`refrac_steps = int(round(REFRAC_MS / DT))`). That contrast taught me the real difference between a mechanistic model and a phenomenological one. The F–I curve made the cost of LIF's simplicity visible: its rate saturated in steps (20, 40, 60 … 180 Hz) because of the hard 5 ms refractory rule and the 50 ms counting window.

---

### Evidence 3 — Lab 7: priors move decision boundaries; populations beat noise

**What I produced.** A Bayesian decoder for three movement directions (priors Left 0.2, Forward 0.5, Right 0.3), a Gaussian tuning-curve fit, and a population-vector decoder with sweeps over neuron count and noise.

| Spike count R | P(Left) | P(Fwd) | P(Right) | MAP |
|---|---|---|---|---|
| 4 | 0.736 | 0.263 | 0.001 | Left |
| 6 | 0.475 | **0.516** | 0.008 | Forward |
| 10 | 0.080 | **0.800** | 0.120 | Forward |
| 14 | 0.005 | 0.418 | **0.577** | Right |

| Neurons N | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| Mean abs. error (°) | 2.57 | 1.87 | 1.34 | 0.95 | 0.66 | 0.47 |

`[Insert Screenshot of Bayesian decoder likelihoods/posteriors from Week 7 here — portfolio_figures/W7_bayesian_decoder.png]`

`[Insert Screenshot of population-vector error vs N and vs noise from Week 7 here — portfolio_figures/W7_popvector_sweeps.png]`

**Reflection.** With equal priors the Left/Forward boundary would sit halfway between the likelihood peaks, at 7.5. Solving the posterior equality with the real priors moved it to **5.85**. The prior literally widened Forward's territory. Even at the most favourable count (R = 10), the decoder was only 80% confident, because overlapping likelihoods (σ = 3) leave real ambiguity. The population-vector sweep reproduced the $1/\sqrt N$ law I had only read about: 8 → 128 neurons gave about a 4× smaller error. A test I added myself, clustering all preferred directions in 0–90°, raised the error to about 40° and left 7 of 24 directions undecodable. That showed me the textbook population vector silently assumes uniform coverage.

---

### Evidence 4 — Lab 6: Hebbian learning is unstable until you normalise it

**What I produced.** I trained a Hebbian neuron on correlated 2-D inputs, compared it with Oja's rule over 30 epochs, and built a 16-unit associative memory tested against noise (300 random cues per pattern).

| After 30 epochs | ‖w‖ | Direction |
|---|---|---|
| Basic Hebbian | **2.1 × 10⁸** | [0.717, 0.697] |
| Oja's rule | **1.0003** | [0.718, 0.696] |
| First principal component | — | [0.712, 0.703] |

`[Insert Screenshot of Hebbian vs Oja weight norm (log scale) from Week 6 here — portfolio_figures/W6_hebb_vs_oja_norm.png]`

`[Insert Screenshot of recall accuracy vs noise (averaged) from Week 6 here — portfolio_figures/W6_recall_vs_noise_averaged.png]`

**Reflection.** Plain Hebbian learning grew about 2.06× per epoch and never stopped. It is a positive-feedback loop with no damping, so the weights can only grow. Oja's single extra term fixed the magnitude while keeping the *direction*, which was exactly the first principal component. This lab directly prepared me for the assignment: the SNN's **weight normalisation** (each neuron's incoming weights rescaled to sum to 78) is the same idea in a different form. The memory experiment showed graceful degradation: perfect recall rate fell from 100% at 0% noise to 0% at 50% noise. At 50% noise the cue is equally close to the pattern and its inverse, so the network has nothing to go on.

---

### Other labs at a glance

| Week | Lab | One-line takeaway |
|---|---|---|
| 1 | EEG rhythms (PhysioNet EEGBCI) | Alpha power was highest at O2/O1/Iz/Oz (posterior), as theory predicts. Mean alpha (9.45e-12) exceeded mean beta (3.91e-12). |
| 3 | Spike trains and decoding | Independent noise averages out with N (0.81 → 0.99 accuracy), but **shared noise** cannot (it reached only 0.94 at N = 50). A ±5 ms coincidence detector fell from 100% to 38% detection at 10 ms jitter. |
| 5 | Izhikevich model | Changing only *a, b, c, d* gave regular (23 Hz, adapting ISI 23.7 → 45.1 ms), fast (131 Hz) and bursting (87 Hz) neurons. |
| 9 | Information theory | MI fell from 0.43 to 0.05 bits as noise SD went from 0.5 to 2.0. Changing bins from 10 ms to 100 ms collapsed two different timing patterns into the same count (1 bit → 0 bits). |

---

## 3. Assignment Insight — Spiking Neural Network for MNIST

*Source: `IT5092_Assignment_SNN_MNIST/MS26910222_IT5092_Assignment.ipynb`*

### 3.1 The problem

I had to replicate the **Diehl & Cook (2015)** architecture in **Brian2**: a spiking network that learns to recognise handwritten MNIST digits **without labels or backpropagation**, using only spike-timing-dependent plasticity, lateral inhibition and homeostasis. Labels are used only *after* training, to name what each neuron has learned.

```
784 Poisson inputs ──(plastic STDP, all-to-all)──► 400 excitatory LIF (adaptive θ)
                                                     │ i==j            ▲ i≠j
                                                     ▼                 │
                                                400 inhibitory LIF ────┘
```

### 3.2 Modelling choices and why I made them

| Component | What I implemented | Why |
|---|---|---|
| **Input encoding** | Pixel → Poisson rate, `rate = pixel/8 × 2` (max ≈ 63.75 Hz) | Rate coding (Week 3): intensity becomes firing rate |
| **Excitatory neurons** | Conductance-based LIF, τ_m = 100 ms, adaptive threshold θ (+0.05 mV per spike, τ_θ = 10⁷ ms) | θ is intrinsic plasticity: frequent winners make themselves harder to fire, which forces specialisation |
| **Inhibitory neurons** | Faster LIF (τ_m = 10 ms), no θ | They only relay the "winner" signal, so they must react quickly |
| **Integration method** | Changed `euler` to **`exponential_euler`** | Forward Euler became numerically unstable under strong inhibition (see Section 4) |
| **STDP (triplet)** | On pre: `ge += w`, `pre = 1`, LTD `w -= 0.0001·post1`. On post: snapshot `post2before = post2`, LTP `w += 0.01·pre·post2before`, reset traces | Snapshotting `post2` before resetting it preserves timing dependence; otherwise LTP would always multiply by 1 |
| **Conductance units** | `ge`, `gi` dimensionless; synapses add plain numbers, e.g. `ge_post += W_EXC_TO_INH` | Brian2 raised `DimensionMismatchError` with `*mV`. The unit system acted like a type checker |
| **Winner-take-all** | Exc→Inh one-to-one (`i == j`, w = 10.4); Inh→Exc lateral (`i != j`, w = 17) | The winner is not punished, only its competitors |
| **Training loop** | Normalise weights (column sum = 78) → present 350 ms → rest 150 ms → if fewer than 5 spikes, raise intensity and retry (up to 5 times) | Normalisation prevents runaway potentiation (Lab 6 again). Rest lets traces decay between images |
| **Testing** | STDP disabled via the synapse namespace (`LTP_RATE = LTD_RATE = 0`) | I verified it: **max weight change during test = 0.00e+00** |

### 3.3 Results

| Metric | Value |
|---|---|
| Training set / test set | 1,000 / 50 images |
| Training accuracy (running) | 63.6% |
| **Test accuracy** | **64.0%** (chance = 10%) |
| Mean excitatory spikes per image | ~17.6 (out of 400 neurons) |
| Intensity boosts needed | 21 (over 1,000 images) |
| Silent neurons / neurons with fewer than 5 spikes | 8 / 84 |
| Share of all spikes from the top 25 neurons | 43.1% |
| Neurons per digit class | 15 (digit 1) to 66 (digit 2) |
| Training wall-clock time | 11.8 min (NumPy backend) |

Per-class test accuracy was best for **1 and 6 (100%)** and worst for **4 (33%) and 9 (43%)**. 4 and 9 share similar strokes.

`[Insert Screenshot of training accuracy curve from Assignment here — portfolio_figures/A2_snn_training_accuracy.png]`

`[Insert Screenshot of excitatory training raster from Assignment here — portfolio_figures/A3_snn_training_raster.png]`

`[Insert Screenshot of receptive fields of the 25 most active neurons from Assignment here — portfolio_figures/A5_snn_top25_receptive_fields.png]`

**What the results mean.** The receptive-field plot is the most convincing evidence. Without ever seeing a label, the most active neurons grew clean prototypes of 0, 1, 3, 5, 6, 7, 8 and 9. STDP did this because when a neuron wins an image, the stroke pixels have *just* fired, so their pre-traces are high and only those synapses get potentiated. The raster's horizontal streaks show θ at work: a neuron wins a run of images, its threshold climbs, and it hands over to others. The provided 0–1 colour scale made most fields look like noise, because after normalisation the mean weight is only about 0.1 (max 0.46). I added a per-neuron colour scale to see what was really there. That is a reminder that a visualisation choice can hide a correct result.

Training and test accuracy were almost identical (63.6% vs 64.0%), so the network is **under-trained, not overfitting**. The median neuron fired only 10 times in the whole run.

### 3.4 What I would improve next

1. **Train on far more data.** Diehl & Cook used the full 60,000 images (about 83% with 100 neurons, about 95% with 6,400). Biologically, STDP is meant to be slow and incremental.
2. **Make that feasible computationally.** At 11.8 min per 1,000 images, 60,000 images would take about 12 hours on the NumPy backend. I would switch Brian2 to `cpp_standalone` code generation, or a GPU backend (Brian2GeNN/Brian2CUDA). I would also run several independent seeds in parallel processes rather than one long run.
3. **Assign labels with frozen weights.** I assigned neuron labels from spike counts recorded *during* training while the weights were still changing. A cleaner protocol is: train → freeze → present a labelled assignment set → assign → test.
4. **Measure more honestly.** With only 50 test images, one image is worth 2%. The standard error is about ±6.8%, so a 95% interval is roughly 51–77%. I would test on at least 1,000 images, report mean ± SD over several seeds, and add a confusion matrix to confirm the 4↔9 and 7↔9 confusions.

---

## 4. Mistakes and Debugging

### Main case: a "numerical seizure" in the SNN (forward-Euler instability)

**The symptom.** Training ran normally for the first ~80 images. Then the network went into a runaway state. **All 400 excitatory neurons fired at their refractory limit** for the rest of training: about 36,400 spikes per 500 ms window instead of about 15. Learning stopped making sense: when every neuron fires for every image, no neuron can specialise.

**First clue: a violated physical invariant.** When I inspected the state variables, I found `v` at **−341 mV** and `gi` in the thousands. A conductance-based neuron can never go below its most negative reversal potential, which here is **E_inh = −100 mV**. The equation drives `v` *towards* −100 mV, never past it. So this was not a modelling or biology problem; it was a **numerical** problem. Finding a value that "cannot happen" is useful in any debugging: it tells you which layer the bug lives in.

**Root cause.** Brian2's `method='euler'` applies the update

$$v_{n+1} = v_n + \Delta t \cdot \frac{-(1 + g_e + g_i)\,v_n + \dots}{\tau_m}$$

The effective decay rate is $k = (1+g_e+g_i)/\tau_m$. Forward Euler is stable only if $\Delta t \cdot k < 2$. With Δt = 0.5 ms and τ_m = 100 ms, that requires

$$1 + g_e + g_i < 400.$$

One inhibitory spike adds 17 to `gi`. So if about **24 excitatory neurons fire in the same 0.5 ms step**, every other neuron gets gi ≈ 24 × 17 > 400, and the update overshoots. The overshoot oscillates with growing amplitude: `v` swings far below −100 mV, then above threshold. That causes more spikes, which add more inhibition, which causes bigger overshoots: a positive-feedback loop. Early in training the receptive fields are similar, so many neurons do fire together at stimulus onset. That is why it only appeared after ~80 images, once a few neurons had learned overlapping patterns.

**The constraint.** The obvious fix, reducing `DT`, was not allowed: the assignment rules say `config.py` must not be changed. So I had to fix the *integrator*, not the *step size*.

**The fix.** I switched both neuron groups to **`method='exponential_euler'`**. For an equation that is linear in `v` once the conductances are fixed, exponential Euler solves each step exactly:

$$v_{n+1} = v_\infty + (v_n - v_\infty)\,e^{-k\Delta t}$$

The factor $e^{-k\Delta t}$ is always between 0 and 1, so `v` decays *towards* its target and can never overshoot a reversal potential, however large the conductance. All parameter values stayed unchanged.

**Verification.** After the fix: mean spikes per image stayed stable at about 17 for all 1,000 images, `v` stayed within its physical bounds, only 21 intensity boosts were needed, and the network learned digit-shaped receptive fields. I documented the reasoning in a Markdown note directly below the cell, so a reader or examiner can follow the decision.

**What I learned.**

- A model can be **mathematically correct and numerically wrong**. The equations matched Diehl & Cook exactly; the integrator did not respect their stiffness.
- **Physical invariants are free assertions.** "v ≥ E_inh" works like a runtime assert. If I had monitored `min(v)` from the start, I would have caught the problem at image 80 instead of after a long run.
- This mirrors control systems: too large a step (gain) on a stiff system produces oscillation, not just inaccuracy.

`[Insert Screenshot of the integration-method note / runaway raster (if saved) from Assignment here]`

### Smaller bugs worth recording

| Where | Error or weak result | Cause | Resolution |
|---|---|---|---|
| Assignment, TODO 1 | `NameError: E_INH` | Template used a name that did not exist in `config.py` (the real names are `E_INH_E` / `E_INH_I`) | Used the correct config constant. Lesson: `config.py` is the single source of truth |
| Assignment, TODO 4 | `DimensionMismatchError` on `ge_post += W*mV` | `ge`/`gi` are dimensionless, so adding millivolts is a unit error | Removed `*mV`. Brian2's unit system behaved like a static type checker |
| Lab 6 | Hebbian weights never changed | With w = 0, the output y = w·x = 0, so Δw = η·x·y = 0 forever (a trivial fixed point) | Initialised with small random weights in [0, 0.1] and added a check proving w = 0 stays at 0 |
| Lab 1 | Peak frequency at O1 reported as **1.02 Hz**, not in the alpha band | `argmax` over a 1/f spectrum always picks the lowest frequency, and the eyes-open run has weak alpha | Read the band power and the plot instead of a raw argmax. Next time: search for a peak *within* 8–12 Hz relative to the 1/f trend, or compare against the eyes-closed run |
| Lab 4 | Empirical LIF rheobase 2.0 vs analytical 1.5 | At I = 1.5, V approaches threshold only asymptotically and never crosses it in finite time | Understood that the analytical value is a limit; the sweep resolution also hid values between 1.5 and 2 |

---

## 5. Comparison and Synthesis

### 5.1 Biological neurons vs computational neuron models

| Aspect | Biological neuron | Hodgkin–Huxley | LIF | Izhikevich |
|---|---|---|---|---|
| Spike generation | Voltage-gated Na⁺/K⁺ channels | Explicit channel kinetics (m, h, n) | Threshold crossing plus reset | Quadratic v term plus reset |
| Spike shape | Full waveform | Full waveform (≈ +40 mV peak) | No shape, just an event | Approximate shape |
| Refractoriness | Na⁺ inactivation, K⁺ afterhyperpolarisation | **Emerges** (≈ 11 ms in my lab) | Hand-coded rule | Via recovery variable u |
| Firing patterns | Many (adaptation, bursting, …) | Depends on channels | Tonic only | Many, from 4 parameters |
| Cost per neuron | — | 4 ODEs, small Δt (0.01 ms) | 1 ODE, large Δt | 2 ODEs, moderate Δt |
| Best used for | — | Understanding mechanisms | Large networks (e.g. my SNN) | Large networks with realistic dynamics |

**My synthesis:** model choice is an engineering trade-off between fidelity and scale. You choose the cheapest model that preserves the phenomenon you care about. For 800 neurons and 313,600 plastic synapses, LIF was the right choice. For explaining *why* refractoriness exists, only HH would do.

### 5.2 Encoding vs decoding

| | Encoding | Decoding |
|---|---|---|
| Question | Stimulus → response | Response → stimulus |
| Probability | $P(R\mid S)$ | $P(S\mid R)$ |
| My examples | Gaussian tuning curve (R² = 0.9965), sigmoid rate code | Bayesian MAP decoder, population vector, linear decoder |
| Main use | Understanding *representation* | Measuring *usable information*; BCIs |
| Link | Bayes' rule turns an encoding model plus a prior into a decoder | |

### 5.3 EEG vs other neuroimaging approaches

| | EEG | fMRI | Structural MRI |
|---|---|---|---|
| Signal | Summed postsynaptic potentials | BOLD (blood oxygenation) | Proton relaxation (T1/T2) |
| Directness | Direct (electrical) | Indirect (haemodynamic) | Not activity at all |
| Temporal resolution | Milliseconds | Seconds (HRF lag) | Not applicable |
| Spatial resolution | Poor (volume conduction; dipoles merge, as in Lab 2) | Millimetres | Sub-millimetre |
| Typical analysis | PSD, ERPs, time–frequency, ICA | GLM, functional connectivity, MVPA | Morphometry, cortical thickness |
| Question answered | **When?** | **Where?** | **What does it look like?** |

Combining modalities, for example simultaneous EEG–fMRI with single-trial ERP amplitudes used as regressors in the fMRI GLM, is how researchers get both *when* and *where*.

### 5.4 Feedforward vs recurrent network behaviour

| | Feedforward | Recurrent |
|---|---|---|
| Information flow | One direction, input → output | Loops; output feeds back |
| Dynamics | Output is a fixed function of the current input | State evolves over time; settles into attractors |
| My examples | Lab 3 linear decoder; the input → excitatory layer of the SNN | Lab 6 associative memory (`x ← sign(Wx)`); the SNN's excitatory ↔ inhibitory loop |
| Strength | Simple, stable, fast | Pattern completion, memory, competition (winner-take-all) |
| Risk | Cannot "clean up" a noisy input | Spurious attractors, runaway dynamics (my Euler bug *was* runaway recurrent inhibition) |

The SNN combines both: a **feedforward** pathway carries the image in, and a **recurrent** inhibitory loop makes the neurons compete. My most useful insight was that the recurrent loop is where both the computation (winner-take-all) *and* the instability (the numerical seizure) lived.

### 5.5 Hebbian learning vs STDP

Hebbian learning detects **correlation** ("fired together"). STDP detects **causation** ("fired just before"). The extra LTD side of STDP adds competition between synapses. Together with weight normalisation and θ, that is what turned a correlation detector into a digit-prototype learner in my assignment.

### 5.6 Putting it together

The thread running through this module is that **the same chain of reasoning applies at every scale**:

> **Biology** (ion channels, synapses, populations) → **equations** (HH, LIF, $1/r^2$, Bayes, entropy) → **simulation** (Python, Brian2) → **data** (spike counts, PSDs, receptive fields) → **interpretation** (what does the number mean, and what could make it wrong?)

The EEG labs went from synchrony physics to measured alpha power. The coding labs went from Poisson spikes to decoding accuracy. The assignment went from STDP biology to 64% unsupervised digit recognition. In each case, the most important step for me was the last one: asking whether a result reflects the brain, the model, or the numerics.

---

## 6. Concepts I Am Still Consolidating

Following the module guidance, this is my honest running list for exam and viva revision:

- [ ] Deriving the HH α/β rate functions' shapes from voltage-clamp data, beyond just using them.
- [ ] Multiple-comparison correction in fMRI (FWE vs FDR vs cluster-based thresholding): when to use which.
- [ ] HRF convolution in practice: how the choice of canonical HRF affects GLM sensitivity.
- [ ] Why the triplet STDP rule specifically improves on pair-based STDP for rate-coded inputs.
- [ ] Estimating entropy and MI with limited data: bias-correction methods for small samples.
- [ ] The MR physics behind T1 vs T2 weighting, beyond the B0/Larmor basics.

---

*All results quoted above come from my own notebooks and lab submissions (Weeks 1–9 and the SNN assignment). Figures are in `portfolio_figures/`.*
