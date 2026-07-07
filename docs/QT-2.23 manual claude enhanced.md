# QT-2.23 — Quantum Sensor Fusion & Stealth Detection
## MASTER BUILD MANUAL v2 — Combined, Audited, and Evaluation-Criteria-Optimized

This document merges your team's original build manual with the advanced techniques your ideation 2 proposed, after independently checking each proposal for mathematical correctness, checking BQPhy's actual published SDK behavior against what was assumed, and re-deriving anything that didn't hold up. Where face3 was right, that fix is now the default. Where face3 was right but the fix is too risky for hackathon timelines, it's marked **[STRETCH]** with the safer default kept as primary. Where face3 overstated something, that's called out explicitly so you don't repeat it in front of judges.

Read Sections 0–3 once, start to finish. Then build module by module (Sections 4 onward), checking off each validation checklist before moving to the next module.

---

## 0. Two Decisions, Made For You

**On the "multi-LLM system" idea:** Don't build it. Not because it's a bad idea in general — it's that it doesn't map to any line item in the evaluation rubric (Methodology 30%, Detection Performance 25%, QuantumNow Usage 20%, Contribution Analysis 15%, Documentation 10%). A multi-agent LLM orchestration layer adds engineering surface area, new failure modes, and nothing a judge will score you on — it actively dilutes your QuantumNow story, which is 20% of your grade and the thing you're least likely to nail without focus. Your own instinct ("too many cooks spoil the dish") was correct. The one place an LLM legitimately helps is invisible to the judges and low-risk: using a coding assistant (Claude, etc.) to help you *write* the pipeline faster, or to help draft your slide narration — that's tooling, not a system component, and it needs zero mention in your architecture diagram. If you want a genuinely novel addition instead, see Section 3.4.

**On face3's suggestions:** Mixed bag — some are real upgrades, one is solving a problem you don't actually have, and the code samples contain a few bugs that would have cost you in front of judges. Full breakdown in Section 2. Net effect: this manual is stronger than either source document alone.

---

## 1. Evaluation Criteria Map — Read This Before You Write a Line of Code

Every module below is tagged with which rubric line item it primarily serves. Use this table to triage your remaining days if you run short on time — never let Documentation (10%) eat time that Methodology (30%) or Detection Performance (25%) needs.

| Criterion | Weight | Primarily served by | Secondarily served by |
|---|---|---|---|
| Soundness & Effectiveness of Sensor Fusion Methodology | 30% | Modules A, B, C (real physics, not noise+label), E (CRLB done honestly), H (Rayleigh-LDA derivation), H.6 (knowing what *not* to quantize) | F (honest baselines), O (avoiding the pitfalls) |
| Detection Performance, esp. weak/low-signature targets | 25% | Modules A–C (calibrated to make stealth genuinely hard), F, G, H (the actual fusion+optimization result), K (so the number is trustworthy) | D (features that let weak signal survive into the classifier) |
| Effective Use of QuantumNow & Optimization Strategy | 20% | Modules G, H, I, J (the actual QUBO/continuous formulations, SDK integration, and scaling argument) | E (CRLB-informed priors), N (backup plan so a live outage doesn't kill this 20%) |
| Quality of Sensor Contribution & Feature Analysis | 15% | Module L (three independent methods, shown agreeing) | Module M (contribution analysis *under degradation* is a strong bonus angle here) |
| Documentation, Reproducibility, Presentation | 10% | This entire manual's discipline: fixed seeds, fixed hardware parameters, validation checklists, honest negative-result handling (Module N) | Section 2 (showing you audited your own claims is itself a documentation/soundness signal) |

**The single biggest scoring lever:** 30% (Methodology) + 25% (Detection) + 20% (QuantumNow) = 75% of your grade lives in Modules A–J. Contribution analysis and documentation are real but secondary. Build in that order.

---

## 2. Accuracy Audit of face3's Proposed Improvements

You asked specifically not to trust this blind. Here's the verdict on each claim, with reasoning — not just "yes/no."

### 2.1 "CRLB falsely inflates quantum efficacy under Weibull clutter" — **Partially correct, conclusion overstated**

face3 is right that the Cramér–Rao Lower Bound assumes regularity conditions (smooth, log-concave-ish likelihood near the true parameter) that genuinely strain under heavy-tailed, low-SNR clutter. That's a real, textbook-documented limitation of CRLB (Kay, *Fundamentals of Statistical Signal Processing*, Ch. 3).

But the conclusion — "if you rely on CRLB as your baseline, you'll falsely inflate the quantum approach's apparent efficacy" — doesn't actually apply to *this* architecture, because **CRLB was never your comparison baseline in the first place.** Look at Module F.5: your real apples-to-apples comparison is *classically-optimized weights solving the exact same Rayleigh-LDA objective QIEO solves* — same objective, same constraints, only the solver differs. CRLB only ever served two narrow roles: (1) a closed-form sanity-check baseline (Module F.2, the "no optimizer at all" floor) and (2) an initialization prior for the optimizer (Module E.3). Neither role is "the thing QIEO has to beat to prove quantum advantage" — Baseline 4 is.

**What's genuinely worth adopting from this critique:** don't present the theoretical CRLB number as if it's an achievable performance bound under your actual (non-Gaussian) clutter. Module E below now reports CRLB *alongside* an empirical Monte Carlo estimate of estimator variance under your real clutter distribution, and explicitly notes the gap. This is a strict upgrade in rigor at near-zero extra build cost — keep it.

### 2.2 "mRMR-to-QUBO via Dinkelbach is mathematically broken with a heuristic solver" — **Correct concern, but it's solving a problem you can opt out of**

This is the most technically sophisticated point face3 raised, and the math is right: Dinkelbach's algorithm (Dinkelbach, 1967) provably converges superlinearly *only if each sub-problem is solved exactly*. QIEO, like every quantum-inspired/annealing solver, is a heuristic — it does not guarantee the global optimum on a single call. Feed an inexact solution into the Dinkelbach update and the sequence parameter λ_k can fail to converge monotonically. This is a legitimate, citable concern (it's the standard "inexact fractional programming" problem in the optimization literature).

**Here's the part face3 missed:** the original mRMR paper (Peng, Long & Ding, 2005, *IEEE TPAMI*) defines **two** equally standard criteria — Mutual Information Quotient (MIQ, the ratio Rel/Red that needs Dinkelbach) **and** Mutual Information Difference (MID, the subtraction Rel − Red). Both are "real mRMR," not an invented shortcut. MID is directly QUBO-compatible with **zero** fractional-programming machinery, **zero** convergence risk, and is brute-force checkable on a 12-feature problem in milliseconds. You are not taking a shortcut by using MID — you are choosing the standard variant that doesn't require an iterative outer loop on top of an already-heuristic inner solver.

**Verdict / what to build:** MID-QUBO is your **default** (Module G, unchanged structure from the original plan, fully validated against brute force). Accelerated-Dinkelbach MIQ is offered as an explicit **[STRETCH]** in Module G.6 *only if* Modules A–L are done early — it is a genuinely more sophisticated technique and, done correctly with the damping fix face3 proposed, is a strong differentiator slide. Just don't let it block your core deliverable, and don't present it without empirically verifying convergence on your own data rather than trusting the theoretical superlinear-convergence claim (which, per the above, doesn't strictly hold here anyway).

### 2.3 "Binary expansion of fusion weights causes catastrophic dimensionality explosion" — **Overstated at your scale, but the proposed fix is real and good**

At 3 sensors × 4 bits/weight = 12 binary variables, this is not remotely catastrophic — D-Wave-class annealers and quantum-inspired solvers routinely handle thousands of binary variables. "Exponential blowup" is a fair description of what happens if you push *resolution* (bits per weight) very high to get many decimal places of precision, but a hackathon demonstration doesn't need that precision, and your own scaling experiment (Module J) already exists specifically to show how solve behavior changes as problem size grows — the bit-resolution sweep is a *feature* of your scaling story, not a flaw to route around.

**That said — the fix face3 proposes is genuinely the better engineering choice, just for a different reason than stated.** I verified directly against BQP's published Python SDK and MATLAB toolbox listings: BQPhy **does** natively support both Continuous and Binary optimization problem types (confirmed via BQP's own MATLAB File Exchange listing, which states the toolbox "supports... Continuous and Binary Optimization problems"). If that's true of the Python SDK too (very likely, same backend), then skipping manual binary encoding entirely for the fusion-weight problem isn't a workaround for a blowup that wasn't really happening — it's simply *less code, fewer places to introduce bugs, and a cleaner story on stage* ("we used the solver's native continuous mode for a continuous problem and its native binary mode for a combinatorial problem — each formulation matches the tool to the problem").

**Verdict / what to build:** Module H now offers the BQPhy native-continuous solve as the **primary** path for fusion weights (Section 9 below), with the fully-derived, brute-force-validated binary-QUBO version (from your original manual, unchanged — it's correct) kept as the **certain fallback** if continuous mode behaves unexpectedly or the exact API doesn't match what's documented publicly. Never go into the event with only one of these working.

The fancier "iterative momentum-biased Rayleigh descent over sequential small QUBOs" that face3 also proposed as an alternative fix is real (this kind of QUBO-based eigenproblem/Rayleigh-quotient descent does appear in published quantum-computing literature on eigenvector solving), but it is meaningfully more complex to implement correctly than either of the two options above, with the highest bug-risk-to-benefit ratio of anything proposed. **Skip it** unless you finish everything else with days to spare.

### 2.4 BQPhy code samples — **Real function names, but several actual bugs found**

I checked BQP's own marketing/SDK pages directly. `bqphy.login()` and `bqphy.solve()` are confirmed as the real top-level integration pattern BQP documents publicly (Step 02/Step 03 of their official onboarding flow). That's a genuine improvement over the fully-generic placeholder pattern in the original manual — use these real names. The exact class names (`OptimizationModel`, `set_quadratic_matrix`, `add_variable`, `problem_type=...`) could not be independently verified — they sit behind a login-gated docs portal (`dev-docs.bosonqpsi.com`) I don't have access to. Treat them as *plausible, not confirmed* until your team logs in and checks Day 1.

More importantly, the actual Python snippets face3 provided contain real bugs that would not run as-is. Fix these before you build on top of them:

| Bug | Where | Fix |
|---|---|---|
| `freqs_sq = 1.0` overwrites the *entire* frequency grid array with a scalar, not just the DC bin — this silently disables all 1/f spectral shaping | Thermal sensor 1/f noise generator | `freqs_sq[freqs_sq == 0] = 1e-6` (only patch the zero/DC entry) |
| `features =` with nothing after it is a syntax error (missing `[]`) | Feature extraction function | `features = []` |
| `enumerate()` called with no argument is a runtime error | SHAP-style contribution function | `enumerate(feature_names)` where `feature_names` is an actual list you define |
| `range(A_scatter.shape)` — `.shape` is a tuple, `range()` needs an int | Continuous solver weight-bounds setup (used twice) | `range(A_scatter.shape[0])` |
| Target signal injected into Weibull clutter map without normalizing to the clutter's local amplitude scale — units don't match between the SNR-derived signal and the unscaled `np.random.weibull(shape, size)` call (which always uses scale = 1) | Radar Weibull clutter generator | See Module A.3 below for the corrected, unit-consistent version |
| CA-CFAR threshold formula `α = N·(Pfa^(−1/N) − 1)` is only valid for **exponential-power** (i.e., Rayleigh-amplitude) clutter — it is silently reused on Weibull-distributed clutter where it is not exact | Radar CFAR detector | See Module A.4 below for the honest fix (Monte Carlo-calibrated threshold) |

None of these are fatal, and most are the kind of thing that happens when someone (human or AI) writes a lot of code fast without running it. But "is this a known method or did you make it up" cuts both ways — a judge who asks you to walk through your CFAR derivation and finds you applied a Rayleigh-only formula to Weibull clutter without comment will mark you down on Soundness (30%) exactly where you most need credit. Fixed versions are in the module sections below.

### 2.5 "QIEO is 3.9x–25x faster than classical methods" — **Real numbers, wrong context — do not cite as your own result**

These multipliers are genuine published figures from BQP's own marketing/benchmark blog posts, but they describe BQP's internal benchmarks on *their* problem instances (CFD, trajectory optimization, generic non-convex structural problems) — not your sensor fusion QUBO. Citing a vendor's general performance marketing as if it validates your specific implementation is exactly the kind of unearned credibility a PhD judge will probe and deflate in one question ("did you measure that, or is that from their website?"). 

**Verdict:** Mention BQP's published benchmarks only as motivating context for *why you chose this tool* ("the vendor reports structural advantages on dense non-convex problems, which matches our fusion-weight landscape's shape"), never as evidence for *your* result. Your Module J scaling experiment is the only legitimate source of a performance claim about your system. If your own numbers come in below the vendor's general claims, report that honestly — see Module N.

---

## 3. Final Recommended Architecture

```
                      ┌─────────────────────────────────────────────┐
                      │   PHYSICS-GROUNDED SYNTHETIC DATA (A,B,C)    │
                      │   Radar (range eqn) · Thermal (Stefan-       │
                      │   Boltzmann) · Acoustic (sonar eqn + LOFAR)  │
                      └───────────────────┬───────────────────────--┘
                                          │
                      ┌───────────────────▼───────────────────────┐
                      │   FEATURE EXTRACTION (D) — ~20 interpretable │
                      │   features across 3 modalities               │
                      └───────────────────┬───────────────────────--┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
  ┌───────────────────────┐   ┌───────────────────────┐   ┌─────────────────────────┐
  │ E: Classical Fisher    │   │ F: Four classical      │   │ K: Reliability protocol │
  │ Info / CRLB + Monte    │   │ baselines (incl. the   │   │ (k-fold, multi-seed,    │
  │ Carlo empirical check  │   │ apples-to-apples one)  │   │ bootstrap CI)           │
  └───────────┬───────────┘   └───────────┬───────────┘   └─────────────────────────┘
              │                           │
              ▼                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  G: QUBO#1 Feature Selection (MID default / Dinkelbach-MIQ       │
  │     stretch) — brute-force validated before any solver call       │
  └───────────────────────────────┬───────────────────────────────--┘
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  H: Fusion Weights — Rayleigh-LDA objective, solved via BQPhy     │
  │     native CONTINUOUS mode [primary] with binary-QUBO fallback    │
  │     [backup] — threshold stays classical (Neyman-Pearson sweep)   │
  └───────────────────────────────┬───────────────────────────────--┘
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  I: BQPhy/QuantumNow SDK integration · J: Scaling experiment      │
  │     (your own numbers, not vendor marketing)                      │
  └───────────────────────────────┬───────────────────────────────--┘
                                  ▼
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
  ┌─────────────────────────┐           ┌─────────────────────────────┐
  │ L: Contribution analysis │           │ M: Stretch — robustness under │
  │ (weights / ablation /    │           │ jamming & degradation         │
  │ SHAP, 3-way agreement)   │           │                               │
  └─────────────────────────┘           └─────────────────────────────┘
```

### 3.1 What changed vs. the original manual
Modules A (clutter realism + honest CFAR), E (CRLB + Monte Carlo), G (MID default + Dinkelbach stretch option), H (continuous-solver primary path), I (real function names), J (vendor-claim discipline).

### 3.2 What's unchanged because it was already correct
B, C, D, F.1–F.4, K, L, M, N, O structure — these modules in the original manual were already sound; face3 didn't materially improve on them, so they're carried forward with only minor notation cleanup.

### 3.3 What was rejected
The multi-LLM orchestration idea (Section 0), the full momentum-biased iterative Rayleigh descent (too much implementation risk for the benefit, Section 2.3), and citing vendor benchmark multipliers as your own performance claim (Section 2.5).

### 3.4 If you want one genuinely novel addition (optional, after everything else is solid)
Not an LLM system. The strongest available "extra" given your rubric is **closing the loop between Module M (degradation) and Module L (contribution analysis)**: show SHAP/ablation contribution values *recomputed live* as each sensor degrades, not just the final detection-rate plot. That's a single new figure, uses code you've already written for both modules, and directly strengthens the 15%-weighted contribution criterion with material no other team is likely to have. This is the "more cooks" version that doesn't spoil the dish, because it's recombination of dishes you're already making, not a new kitchen.

---

## 4. MODULE A — Radar Sensor Simulation

### A.1 The Physics

Monostatic radar range equation (Skolnik, *Introduction to Radar Systems* — the standard reference, cite it exactly like this):

```
            Pt · G² · λ² · σ
SNR  =  ─────────────────────────
         (4π)³ · R⁴ · k · T0 · B · F · L
```

| Symbol | Meaning | Starting value |
|---|---|---|
| Pt | Transmit power (W) | 10,000 W |
| G | Antenna gain (linear) | 1000 (≈30 dBi) |
| λ | Wavelength (m), f = 10 GHz → λ = c/f | 0.03 m |
| σ | Radar Cross Section (m²) — **your stealth knob** | Conventional: 10 m². Stealth: 0.01–0.1 m² |
| R | Range (m) | 50,000 m |
| k, T0 | Boltzmann constant, ref. temperature | 1.380649×10⁻²³ J/K, 290 K (physical constants — never change) |
| B | Bandwidth (Hz) | 1 MHz |
| F | Noise figure (linear) | 3 (≈4.77 dB) |
| L | System losses (linear) | 4 (≈6 dB) |

Decibel form (use in code to avoid floating-point underflow):

```python
import numpy as np

def radar_snr_db(rcs_m2, range_m, freq_hz=10e9, Pt_W=10000.0,
                  G_dB=30.0, B_hz=1e6, F_dB=4.77, L_dB=6.0):
    c = 299792458.0
    wavelength = c / freq_hz
    k, T0 = 1.380649e-23, 290.0
    Pt_dBW = 10 * np.log10(Pt_W)
    rcs_dBsm = 10 * np.log10(rcs_m2)
    thermal_noise_dBW = 10 * np.log10(k * T0 * B_hz)
    return (Pt_dBW + 2*G_dB + 20*np.log10(wavelength) + rcs_dBsm
            - 30*np.log10(4*np.pi) - 40*np.log10(range_m)
            - thermal_noise_dBW - F_dB - L_dB)
```

**Why σ is the stealth knob:** real low-observable platforms achieve σ on the order of 0.001–0.01 m² via shaping and radar-absorbent materials vs. ~10 m² for a conventional fighter — documented, citable engineering fact.

### A.2 Clutter Model — Default vs. Stretch (the corrected version)

**Default — Rayleigh amplitude clutter (exponential power).** Use this as your primary model. It's the standard Central-Limit-Theorem result for many small uncorrelated scatterers, and it has a clean, exact, closed-form CFAR threshold (next section). Build and validate your entire pipeline on this first.

```python
def generate_rayleigh_clutter(num_cells, clutter_power):
    sigma_c = np.sqrt(clutter_power / 2)
    I = np.random.normal(0, sigma_c, num_cells)
    Q = np.random.normal(0, sigma_c, num_cells)
    return np.sqrt(I**2 + Q**2)
```

**Stretch — Weibull clutter (heavy-tailed, more realistic at low grazing angles).** This is the upgrade face3 correctly identified as more representative of real sea/terrain clutter, *if* you handle the unit-consistency and threshold issues correctly (face3's own snippet didn't — see Section 2.4):

```python
def generate_weibull_clutter(num_cells, clutter_scale, clutter_shape=1.5):
    """
    shape (k) = 2.0 reduces to Rayleigh exactly (sanity check this).
    shape < 1.5 produces the heavy-tailed 'spiky' clutter that masks
    stealth targets in low-grazing-angle sea/terrain returns.
    clutter_scale sets the RMS amplitude — calibrate this to match
    the same noise-floor power you use elsewhere in the pipeline, so
    swapping Rayleigh <-> Weibull doesn't silently change your SNR
    definition.
    """
    return clutter_scale * np.random.weibull(clutter_shape, num_cells)

def inject_target_into_clutter(clutter_map, target_idx, snr_db):
    """
    Unit-consistent target injection: scale the injected amplitude by
    the LOCAL clutter RMS, not by an arbitrary constant. This is the
    fix for the bug in face3's snippet, which added a raw SNR-derived
    voltage directly to a Weibull sample of scale=1 with no reference
    to the actual local noise floor.
    """
    local_rms = np.sqrt(np.mean(clutter_map**2))
    target_amplitude = local_rms * (10 ** (snr_db / 20.0))
    out = clutter_map.copy()
    out[target_idx] += target_amplitude
    return out
```

### A.3 CA-CFAR Detector — Default vs. Honest Stretch

**Default (Rayleigh clutter — exact closed form):**

```
T = α · Z,    α = N · (Pfa^(−1/N) − 1)
```

where Z is the mean reference-cell power and N is the number of reference cells. This formula is exact *because* Rayleigh-amplitude clutter has exponentially-distributed power, and the sum of N i.i.d. exponential variables is Gamma-distributed with a known closed form — that's the actual derivation behind this formula, not a coincidence.

```python
def ca_cfar_threshold_rayleigh(num_ref_cells, pfa):
    alpha = num_ref_cells * (pfa ** (-1.0/num_ref_cells) - 1.0)
    return alpha
```

**Stretch — if you've switched to Weibull clutter, you cannot reuse this formula unmodified** (this is the bug face3's code silently committed). General-shape Weibull clutter does not have a simple closed-form distribution for the sum/mean of N reference cells, so the honest, standard engineering fix when no closed form is tractable is **Monte Carlo threshold calibration**:

```python
def calibrate_cfar_threshold_monte_carlo(clutter_scale, clutter_shape,
                                           num_ref_cells, target_pfa,
                                           n_trials=200_000):
    """
    Run the CFAR statistic under H0 (clutter only, no target) many times
    and empirically find the multiplier alpha that hits target_pfa.
    This is a standard, legitimate technique when no closed-form CFAR
    threshold exists for the assumed clutter distribution -- it is not
    a hack, it is what you do instead of misapplying a formula derived
    under a different distributional assumption.
    """
    ref_cells = generate_weibull_clutter(n_trials * num_ref_cells,
                                          clutter_scale, clutter_shape
                                          ).reshape(n_trials, num_ref_cells)
    test_cells = generate_weibull_clutter(n_trials, clutter_scale, clutter_shape)
    Z = ref_cells.mean(axis=1)
    ratio = test_cells / Z          # the CFAR test statistic
    alpha = np.percentile(ratio, 100 * (1 - target_pfa))
    return alpha
```

**Say this explicitly on your slide if you use Weibull clutter:** *"Because Weibull clutter has no closed-form CA-CFAR threshold for general shape parameters, we calibrate the threshold empirically via Monte Carlo to hit our target false-alarm rate exactly — this is standard practice when the regularity conditions for a closed-form CFAR derivation don't hold, and it's the same philosophy we apply to CRLB in Module E."** This sentence alone fixes the exact gap face3's snippet left open, and ties two parts of your presentation together coherently.

### A.4 Validation Checklist for Module A

- [ ] At σ = 10 m², SNR lands ~15–25 dB at baseline range.
- [ ] At σ = 0.01 m², SNR drops ~30 dB vs. conventional (linear σ → 1000× reduction = 30 dB; verify numerically).
- [ ] Rayleigh-CFAR: conventional target Pd > 90% at Pfa = 1e-4.
- [ ] Rayleigh-CFAR: stealth target shows visibly degraded Pd — this gap is what fusion recovers.
- [ ] If using Weibull stretch: shape=2.0 run reduces numerically to the Rayleigh case (sanity check); Monte Carlo-calibrated threshold achieves the target Pfa within ~10% when re-tested on a fresh batch of clutter-only trials.

---

## 5. MODULE B — Thermal/Infrared Sensor Simulation

### B.1 The Physics

Stefan–Boltzmann law (Planck's law integrated over all wavelengths — defensible simplification vs. full band-limited Planck integration, which is a fine stretch goal but not required):

```
M = ε · σ_SB · T⁴        σ_SB = 5.670374×10⁻⁸ W/(m²·K⁴)
```

(Do not confuse σ_SB with radar cross section σ from Module A — separate code variables: `stefan_boltzmann` vs. `rcs`.)

| Parameter | Conventional target | Stealth target | Background |
|---|---|---|---|
| Temperature (K) | 400 | 310 (suppressed) | 290 |
| Emissivity ε | 0.9 | 0.3 (low-ε coating) | 0.95 |

```python
SIGMA_SB = 5.670374e-8

def radiant_exitance(emissivity, temp_K):
    return emissivity * SIGMA_SB * temp_K**4

def thermal_contrast(target_eps, target_T, bg_eps, bg_T):
    return radiant_exitance(target_eps, target_T) - radiant_exitance(bg_eps, bg_T)

def thermal_snr(delta_M, netd_K, eps_avg=0.7, T_ref=290.0):
    # NETD -> noise floor via the local derivative dM/dT
    dMdT = 4 * eps_avg * SIGMA_SB * T_ref**3
    noise_floor = dMdT * netd_K
    return delta_M / noise_floor
```

NETD (Noise-Equivalent Temperature Difference) is the actual spec IR camera manufacturers publish: 50 mK typical for uncooled microbolometers, <20 mK for cooled detectors.

### B.2 1/f Spatial Noise — Corrected Version

Real microbolometer arrays show 1/f (pink) spatial noise, not pure white Gaussian. Here is the **corrected** generator — face3's version had a bug that overwrote the entire frequency grid with the scalar 1.0, which silently disabled all frequency shaping and produced plain white noise while claiming to be 1/f:

```python
def generate_1f_noise(shape, alpha=1.0):
    white = np.random.normal(0, 1, shape)
    f_noise = np.fft.fft2(white)
    fx = np.fft.fftfreq(shape[0])[:, None]
    fy = np.fft.fftfreq(shape[1])[None, :]
    freqs_sq = fx**2 + fy**2
    freqs_sq[0, 0] = 1e-6     # patch ONLY the DC bin, not the whole array
    f_noise = f_noise / (freqs_sq ** (alpha / 2))
    return np.real(np.fft.ifft2(f_noise))
```

Always sanity-plot the power spectrum of your generated noise (`np.abs(np.fft.fft2(noise))**2` on a log-log axis vs. frequency) before trusting it — a straight-ish downward-sloping line confirms the 1/f shaping actually worked; a flat line means you've reintroduced the bug above.

### B.3 Validation Checklist for Module B

- [ ] Conventional target contrast: several hundred W/m² above background.
- [ ] Stealth target contrast: ≥10× reduced vs. conventional (verify the 400K/0.9 → 310K/0.3 drop numerically).
- [ ] Stealth target SNR sits near or below 1 in challenging scenarios.
- [ ] Power-spectrum sanity plot confirms 1/f shaping is actually present (see above).

---

## 6. MODULE C — Acoustic/Sonar Sensor Simulation

### C.1 The Physics

Passive sonar equation: Signal Excess = SL − TL − (NL − DI) − DT. Detection when Signal Excess ≥ 0.

| Symbol | Meaning | Starting value |
|---|---|---|
| SL | Source Level (dB re 1 µPa @ 1m) | Conventional: 160 dB. Stealth (quieted): 110 dB |
| TL | Transmission Loss | Formula below |
| NL | Ambient Noise Level | Knudsen curves, ~60–70 dB |
| DI | Receive array Directivity Index | 10 dB |
| DT | Detection Threshold | 0 dB baseline |

```python
def transmission_loss(range_m, absorption_coef=0.005):
    # Spherical spreading + absorption (valid for short-to-moderate range)
    return 20*np.log10(range_m) + absorption_coef*range_m

def transmission_loss_extended(range_m, absorption_coef=0.005):
    # Spherical spreading near-field, cylindrical spreading beyond 1 km --
    # more realistic for longer-range scenarios; pick one TL model and
    # stay consistent across your whole dataset.
    if range_m < 1000:
        return 20*np.log10(range_m) + absorption_coef*range_m
    return 20*np.log10(1000) + 10*np.log10(range_m/1000.0) + absorption_coef*range_m

def signal_excess(SL_dB, range_m, NL_dB, DI_dB, DT_dB=0.0):
    return SL_dB - transmission_loss(range_m) - (NL_dB - DI_dB) - DT_dB
```

Knudsen/Wenz ambient noise model — wind/sea-state-dependent baseline with the standard −17 dB/decade spectral falloff above ~1 kHz:

```python
def knudsen_noise_level(freq_hz, sea_state=3):
    base_level = 50 + 5 * sea_state
    return base_level - 17.0 * np.log10(max(freq_hz, 1.0) / 1000.0)
```

**Why SL is the stealth knob:** documented quieting technology (anechoic tiles, isolated mounts) is measured directly in source-level reduction — a 30–50 dB SL drop from conventional to modern quiet vessels is open-literature fact.

### C.2 LOFAR Spectrogram Generation

Real passive sonar systems process raw hydrophone time series into a LOFAR (Low-Frequency Analysis and Recording) spectrogram via STFT, then hunt for narrowband tonal lines against broadband ambient noise. Build this explicitly rather than working with raw features only — it gives you a genuine spectrogram feature pool and a strong visual for your slides.

```python
from scipy.signal import stft, get_window

def generate_lofar_spectrogram(signal_excess_db, duration_s=60, fs=2000,
                                 tonal_freqs_hz=(50, 120, 240)):
    t = np.linspace(0, duration_s, duration_s * fs, endpoint=False)
    noise = np.random.normal(0, 1, len(t))
    amplitude = 10 ** (signal_excess_db / 20.0)
    target = sum(amplitude * np.sin(2*np.pi*f*t) for f in tonal_freqs_hz)
    raw_audio = noise + target

    window = get_window('hann', 1024)
    f, t_spec, Zxx = stft(raw_audio, fs=fs, window=window, nperseg=1024, noverlap=512)
    lofar_image = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)
    return {"raw_signal": lofar_image, "snr_db": signal_excess_db,
            "label": 1, "metadata": {"fs": fs, "tonal_freqs": tonal_freqs_hz}}
```

### C.3 Validation Checklist for Module C

- [ ] Conventional target (SL=160 dB) shows strongly positive signal excess at moderate range.
- [ ] Stealth target (SL=110 dB) shows negative/near-zero signal excess at the same range.
- [ ] Tonal lines clearly visible above noise floor in the LOFAR spectrogram for the conventional case; buried/marginal for the stealth case — plot and visually confirm before moving on.

---

## 7. MODULE D — Feature Extraction (shared output contract)

Every sensor module above must produce this exact dictionary so downstream code doesn't care which sensor generated it:

```python
sensor_output = {
    "raw_signal": np.ndarray,   # waveform / image / spectrogram
    "snr_db": float,            # ground-truth SNR used to generate it
    "label": int,                # 1 = target present, 0 = absent
    "metadata": dict             # all physical parameters, for reproducibility
}
```

Reduce raw signals to **interpretable** numeric features — this matters twice: it keeps classifiers tractable, and it's what makes Module L (contribution analysis) actually explainable instead of a black box.

```python
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

def extract_radar_features(range_doppler_map, cfar_threshold_map):
    peak_snr = np.max(range_doppler_map)
    cfar_exceedances = np.sum(range_doppler_map > cfar_threshold_map)
    doppler_spread = np.std(np.argmax(range_doppler_map, axis=1))
    target_to_clutter = peak_snr / np.median(range_doppler_map)
    weibull_skew_estimate = skew(range_doppler_map.flatten())
    return [peak_snr, cfar_exceedances, doppler_spread,
            target_to_clutter, weibull_skew_estimate]

def extract_thermal_features(thermal_frame):
    peak_contrast = np.max(thermal_frame) - np.median(thermal_frame)
    gy, gx = np.gradient(thermal_frame)
    edge_energy = np.mean(gx**2 + gy**2)
    hot_spot_area = np.sum(thermal_frame > (np.median(thermal_frame) + 2*np.std(thermal_frame)))
    contrast_to_noise = peak_contrast / (np.std(thermal_frame) + 1e-12)
    return [peak_contrast, edge_energy, hot_spot_area, contrast_to_noise]

def extract_acoustic_features(lofar_image, tonal_band=(40, 150), fs=2000):
    freqs = np.linspace(0, fs/2, lofar_image.shape[0])
    band_mask = (freqs > tonal_band[0]) & (freqs < tonal_band[1])
    tonal_power = np.sum(lofar_image[band_mask, :])
    broadband_power = np.sum(lofar_image) - tonal_power
    spectral_kurt = kurtosis(lofar_image.flatten())
    tonal_to_broadband = tonal_power / (broadband_power + 1e-12)
    return [tonal_power, broadband_power, spectral_kurt, tonal_to_broadband]
```

This gives 5 + 4 + 4 = 13 baseline features (round to whichever count your team settles on — 12–16 is a sensible range for the feature pool feeding Module G). **Why spectral kurtosis matters:** broadband ambient noise is low-kurtosis (flat spectral shape); tonal/engine components are high-kurtosis (sharp peaks) — this is a real operational LOFAR-analysis feature, not invented for this hackathon.

---

## 8. MODULE E — Classical Fisher Information, CRLB, and Empirical Validation

### E.1 What This Module Is Actually For

**Important, non-negotiable correction:** Quantum Fisher Information (QFI) applies to quantum measurement systems (entangled-photon probes, NV-center magnetometers, squeezed-light interferometers). Radar, thermal cameras, and sonar are classical sensors. Using QFI to justify them is a category error a PhD judge will catch immediately and will cost you directly on Soundness (30%). The correct tool is **classical Fisher Information and the CRLB** — standard estimation theory, fully appropriate here.

```
I(θ) = −E[ ∂² ln L(x;θ) / ∂θ² ]          Var(θ̂) ≥ 1/I(θ)
```

For a Gaussian observation model x ~ N(θ, σ²) (a reasonable approximation for a per-sensor detection statistic under noise): I(θ) = 1/σ². Higher SNR → lower noise variance → higher Fisher Information → tighter achievable estimation error. This gives a principled way to set initial fusion-weight priors.

```python
def crlb_initial_weights(snr_radar_linear, snr_thermal_linear, snr_acoustic_linear):
    fi = np.array([snr_radar_linear, snr_thermal_linear, snr_acoustic_linear])
    return fi / fi.sum()
```

### E.2 The Honest Upgrade: Empirical Monte Carlo Validation Under Your Actual Clutter

Per the audit in Section 2.1, don't present the closed-form CRLB as if it's an achievable bound under your real (possibly heavy-tailed) clutter. Compute both, side by side:

```python
def empirical_estimator_variance(generate_scenario_fn, true_theta, n_trials=2000):
    """
    Monte Carlo estimate of estimator variance under the ACTUAL clutter
    distribution used in your pipeline (Rayleigh or Weibull), rather than
    trusting the Gaussian-regularity assumptions baked into closed-form CRLB.
    """
    estimates = np.array([generate_scenario_fn() for _ in range(n_trials)])
    return np.var(estimates - true_theta)

def report_crlb_vs_empirical(snr_linear, generate_scenario_fn, true_theta):
    theoretical_crlb = 1.0 / snr_linear
    empirical_var = empirical_estimator_variance(generate_scenario_fn, true_theta)
    gap_ratio = empirical_var / theoretical_crlb
    return {"theoretical_crlb": theoretical_crlb,
            "empirical_variance": empirical_var,
            "gap_ratio": gap_ratio}
```

**Report this gap explicitly in your slides under heavy clutter / low-SNR conditions.** A widening gap as SNR drops is itself a scientifically interesting finding ("CRLB underestimates achievable error in the stealth regime, motivating why a combinatorial fusion-weight search matters more there than the closed-form bound alone would suggest") — this is a stronger, more defensible story than presenting CRLB as ground truth.

### E.3 Validation Checklist for Module E

- [ ] For radar SNR ≫ thermal SNR ≫ acoustic SNR, CRLB weights rank radar > thermal > acoustic.
- [ ] Weights sum to 1.
- [ ] Empirical-vs-theoretical CRLB gap computed and plotted across at least 3 SNR levels, including your hardest stealth scenario.

---

## 9. MODULE F — Classical Baseline Detectors

**Your quantum result is only as credible as your classical baselines are honest.** Spend real effort here — a weak baseline that QIEO "beats" is the single fastest way to lose Soundness points.

### F.1 Per-sensor detector (shared by all baselines)

```python
from sklearn.linear_model import LogisticRegression

def train_sensor_detector(X_features, y_labels):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_features, y_labels)
    return clf

def sensor_score(clf, X_features):
    return clf.predict_proba(X_features)[:, 1]
```

Keep this simple deliberately — your innovation lives in the fusion layer, not in exotic per-sensor classifiers.

### F.2 Baseline 1 — CRLB-weighted average (closed form)
```python
def fuse_crlb_weighted(S_r, S_t, S_a, w_r, w_t, w_a):
    return w_r*S_r + w_t*S_t + w_a*S_a
```

### F.3 Baseline 2 — Kalman filter fusion
```python
def kalman_fuse(observations, observation_noise_vars):
    weights = [1.0/v for v in observation_noise_vars]
    total = sum(weights)
    fused = sum(o*w for o, w in zip(observations, weights)) / total
    return fused, 1.0/total
```

### F.4 Baseline 3 — Bayesian sequential fusion
```python
def bayesian_fuse(prior_log_odds, sensor_likelihood_ratios):
    log_odds = prior_log_odds
    for lr in sensor_likelihood_ratios:
        log_odds += np.log(lr)
    return 1.0 / (1.0 + np.exp(-log_odds))
```

### F.5 Baseline 4 — Classically-optimized weights (the critical one)

Solves the **exact same objective** you'll hand to the quantum solver in Module H — this is what makes the QIEO comparison scientifically valid:

```python
from scipy.optimize import minimize, differential_evolution

def fisher_objective(w, S_b, S_w, lam=0.5):
    w = np.asarray(w)
    return -(w @ S_b @ w - lam * (w @ S_w @ w))

def optimize_weights_classical(S_b, S_w, n_sensors=3, n_restarts=10):
    best_result, best_val = None, np.inf
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_sensors
    for _ in range(n_restarts):
        w0 = np.random.dirichlet(np.ones(n_sensors))
        result = minimize(fisher_objective, w0, args=(S_b, S_w),
                           bounds=bounds, constraints=constraints, method='SLSQP')
        if result.fun < best_val:
            best_val, best_result = result.fun, result
    return best_result.x
```

**Run with at least 10 random restarts** — a single-seed classical optimizer that gets stuck in a local optimum and "loses" to QIEO is not a valid finding, it's an under-tuned baseline.

### F.6 Baseline 5 (added per the audit, Section 2.5) — Differential evolution / genetic-algorithm-style global search

A genuinely different classical search strategy than gradient-based SLSQP — useful both as an extra baseline and as the comparator in your scaling experiment (Module J), since it's the closest population-based classical analog to what QIEO is doing:

```python
def optimize_weights_differential_evolution(S_b, S_w, n_sensors=3, lam=0.5):
    bounds = [(0, 1)] * n_sensors
    def objective(w):
        w = np.asarray(w) / np.sum(w)   # project onto simplex via renormalization
        return fisher_objective(w, S_b, S_w, lam)
    result = differential_evolution(objective, bounds, seed=42, maxiter=500)
    return result.x / np.sum(result.x)
```

### F.7 Validation Checklist for Module F

- [ ] All baselines achieve near-perfect detection on a clean/easy scenario.
- [ ] All baselines show degraded but non-zero performance on the stealth scenario.
- [ ] Baseline 4 (and 5) outperform Baselines 1–3 in most scenarios — if not, check optimizer convergence with more restarts before concluding otherwise.

---

## 10. MODULE G — QUBO #1: Feature Selection (mRMR)

### G.1 Default: MID (Mutual-Information-Difference) Formulation — No Fractional Programming Needed

Per the audit (Section 2.2), this is a standard, published mRMR variant (Peng, Long & Ding, 2005) — not a workaround. It's directly QUBO-compatible.

```
H(x) = −α·Σᵢ Relᵢ·xᵢ + β·Σᵢ<ⱼ Redᵢⱼ·xᵢ·xⱼ + γ·(Σᵢ xᵢ − k)²
```

- Relᵢ = |corr(featureᵢ, label)| (or mutual information, if time allows)
- Redᵢⱼ = |corr(featureᵢ, featureⱼ)|
- Third term: quadratic cardinality penalty enforcing exactly k features selected.

Expanding the cardinality penalty (using xᵢ² = xᵢ for binary variables):

```
Q_ii = −α·Relᵢ + γ(1 − 2k)
Q_ij = β·Redᵢⱼ + 2γ      (i ≠ j)
```

```python
def build_feature_selection_qubo(features_matrix, labels, k_target,
                                   alpha=1.0, beta=1.0, gamma=2.0):
    n = features_matrix.shape[1]
    relevance = np.array([abs(np.corrcoef(features_matrix[:, i], labels)[0, 1])
                           for i in range(n)])
    redundancy = np.abs(np.corrcoef(features_matrix.T))
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -alpha*relevance[i] + gamma*(1 - 2*k_target)
        for j in range(i+1, n):
            Q[i, j] = Q[j, i] = beta*redundancy[i, j] + 2*gamma
    return Q
```

**Starting coefficients:** α=1.0, β=1.0, γ=2.0. If solver picks the wrong feature count, increase γ. If it keeps redundant pairs, increase β. If it ignores clearly useful features, increase α or decrease β.

**Always validate against brute force first** (cheap at 12–16 features):

```python
from itertools import combinations

def brute_force_feature_selection(Q, n_features, k_target):
    best_H, best_x = np.inf, None
    for combo in combinations(range(n_features), k_target):
        x = np.zeros(n_features); x[list(combo)] = 1
        H = x @ Q @ x
        if H < best_H:
            best_H, best_x = H, x
    return best_x, best_H
```

Run this *before* touching QuantumNow. A mismatch means your QUBO formulation or solver call has a bug — find it now, not after three more layers are built on top.

### G.2 [STRETCH] MIQ via Accelerated Dinkelbach

If — and only if — Modules A–L are solid with days to spare, the literal mRMR ratio (Relᵢ/Redᵢⱼ, "MIQ") via Dinkelbach's algorithm is a genuinely more sophisticated optional layer:

```python
def solve_mrmr_dinkelbach(relevance, redundancy_matrix, qubo_solver_fn,
                            max_iter=15, tol=1e-4, damping=0.5):
    """
    Standard Dinkelbach iteration with a damping/acceleration term
    (per the audit, Section 2.2) to mitigate non-monotonic lambda
    updates when qubo_solver_fn is heuristic rather than exact.
    """
    lam = 0.0
    x_opt = None
    for k in range(max_iter):
        Q = relevance - lam * redundancy_matrix   # parametric sub-problem
        x_opt, _ = qubo_solver_fn(Q)
        numerator = x_opt @ relevance
        denominator = x_opt @ redundancy_matrix @ x_opt
        if denominator <= 1e-9:
            break
        lam_unconstrained = numerator / denominator
        # Damped update: blend with previous lambda rather than jumping
        # fully to the new value -- this is the practical mitigation for
        # non-monotonic convergence under an inexact (heuristic) solver.
        lam_next = damping * lam_unconstrained + (1 - damping) * lam
        if abs(lam_next - lam) < tol:
            lam = lam_next
            break
        lam = lam_next
    return x_opt, lam
```

**Mandatory if you build this:** empirically plot λ_k across iterations and confirm it actually settles rather than oscillating, on *your* data — don't cite Dinkelbach's theoretical superlinear-convergence guarantee, since that guarantee assumes exact sub-problem solving, which a heuristic QIEO call does not provide (this is the exact gap face3 flagged, and the reason for the damping term above).

### G.3 Validation Checklist for Module G

- [ ] Brute force confirms a unique (or near-unique) minimum-H subset for the MID formulation.
- [ ] QIEO solver result matches brute force on the 12–16 feature case.
- [ ] Selected features make physical sense given the scenario's per-sensor SNR.
- [ ] [If built] Dinkelbach λ-sequence empirically confirmed to converge on your data, not assumed from theory.

---

## 11. MODULE H — Fusion Weight Optimization

### H.1 The Objective (Fisher / Rayleigh-LDA)

```
              (μ1(w) − μ0(w))²
J(w)  =  ─────────────────────────
              σ1²(w) + σ0²(w)
```

A generalized Rayleigh quotient — the same object underlying classical LDA, a century-old, well-validated statistical tool, applied here to a new combinatorial-adjacent problem.

QUBO/QIO solvers minimize polynomials, not ratios directly. The standard, defensible fix is Lagrangian relaxation — maximize the numerator while penalizing the denominator with trade-off λ:

```
J'(w) = (μ1(w) − μ0(w))² − λ·(σ1²(w) + σ0²(w))  =  w^T(S_b − λS_w)w
```

This is a clean quadratic form — falls directly out of the algebra, not forced. **State this explicitly on a slide as a deliberate, justified simplification.** S_b is the between-class scatter matrix, S_w the within-class scatter matrix (standard LDA objects).

```python
def compute_scatter_matrices(sensor_scores_class1, sensor_scores_class0):
    mu1, mu0 = sensor_scores_class1.mean(axis=0), sensor_scores_class0.mean(axis=0)
    diff = (mu1 - mu0).reshape(-1, 1)
    S_b = diff @ diff.T
    S_w = np.cov(sensor_scores_class1.T) + np.cov(sensor_scores_class0.T)
    return S_b, S_w
```

### H.2 [PRIMARY] BQPhy Native Continuous Solve

Per the audit (Section 2.3) — confirmed via BQP's own published toolbox listing that BQPhy supports Continuous as well as Binary optimization problem types — this is the recommended primary path: it matches a continuous problem to a continuous solver, with no manual bit-encoding to debug.

```python
import bqphy   # confirmed real top-level module per BQP's own SDK onboarding docs

def solve_fusion_weights_continuous(S_b, S_w, lam=0.5, n_sensors=3):
    """
    NOTE: bqphy.login() and bqphy.solve() are confirmed real entry points
    from BQP's own public SDK documentation. The exact class/method names
    below (OptimizationModel, add_variable, set_objective, problem_type=
    'Continuous') are the PLAUSIBLE PATTERN based on BQP's documented
    binary-problem flow and their confirmed support for continuous
    problems -- but they sit behind a login-gated docs portal we could
    not directly verify. CONFIRM THESE EXACT NAMES on Day 1 against your
    real BQPhy SDK docs/example notebook before building further on top
    of this function -- see Module I.3, Step 2.
    """
    model = bqphy.OptimizationModel(problem_type="Continuous")
    for i in range(n_sensors):
        model.add_variable(name=f"w_{i}", var_type="Continuous", bounds=(0.0, 1.0))

    def objective(w_dict):
        w = np.array([w_dict[f"w_{i}"] for i in range(n_sensors)])
        w = w / np.sum(w)   # enforce simplex constraint by renormalization inside the objective
        return -(w @ S_b @ w - lam * (w @ S_w @ w))   # negate: BQPhy minimizes by default

    model.set_objective(objective)
    result = bqphy.solve(model, solver="QuantumNOW")
    raw = result.get_continuous_solution()
    w = np.array([raw[f"w_{i}"] for i in range(n_sensors)])
    return w / np.sum(w)
```

### H.3 [CERTAIN FALLBACK] Binary QUBO with Fixed-Point Expansion

Keep this working *before* the event, not as a thing you build only if the continuous path fails live. This is unchanged from the original manual because it was already correct, and it's fully brute-force validated.

```
wᵢ = Σ_{p=1}^{b} 2^(−p)·x_{i,p},   x_{i,p} ∈ {0,1}
```

b=4 bits/weight → resolution 1/16 ≈ 0.0625, plenty for a demonstration. 3 sensors × 4 bits = 12 binary variables.

```python
def weights_from_bits(bits, n_sensors, bits_per_weight):
    w = np.zeros(n_sensors)
    for i in range(n_sensors):
        for p in range(bits_per_weight):
            w[i] += bits[i*bits_per_weight + p] * (2 ** -(p+1))
    return w

def build_weight_qubo(S_b, S_w, lam, n_sensors=3, bits_per_weight=4):
    n_vars = n_sensors * bits_per_weight
    Q = np.zeros((n_vars, n_vars))
    bit_value = lambda p: 2 ** -(p+1)
    M = S_b - lam * S_w
    for i in range(n_sensors):
        for p in range(bits_per_weight):
            idx_ip = i*bits_per_weight + p
            for j in range(n_sensors):
                for q in range(bits_per_weight):
                    idx_jq = j*bits_per_weight + q
                    Q[idx_ip, idx_jq] += M[i, j] * bit_value(p) * bit_value(q)
    return -Q   # negate: maximize J' becomes minimize -J' for the solver

def add_simplex_constraint(Q, n_sensors, bits_per_weight, penalty_weight=5.0):
    Qc = Q.copy()
    bit_value = lambda p: 2 ** -(p+1)
    for i in range(n_sensors):
        for p in range(bits_per_weight):
            idx_ip = i*bits_per_weight + p
            Qc[idx_ip, idx_ip] += penalty_weight * (bit_value(p)**2 - 2*bit_value(p))
            for j in range(n_sensors):
                for q in range(bits_per_weight):
                    if (i, p) != (j, q):
                        idx_jq = j*bits_per_weight + q
                        Qc[idx_ip, idx_jq] += penalty_weight * bit_value(p) * bit_value(q)
    return Qc
```

### H.4 Why the Detection Threshold Stays Classical

The problem statement asks you to optimize "fusion weights, feature selection, and detector hyperparameters." Resist the temptation to push the detection threshold τ into the QUBO too — it's a single continuous scalar setting a Neyman-Pearson operating point, a 1-D sweep, not a combinatorial problem. Forcing it into QUBO form adds complexity without genuine quantum advantage.

```python
from sklearn.metrics import roc_curve

def select_threshold_neyman_pearson(fused_scores, labels, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, fused_scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return thresholds[idx], tpr[idx], fpr[idx]
```

**Say this explicitly on stage:** *"We deliberately kept threshold selection classical because it is not combinatorial — quantum optimization should be reserved for problems with genuine combinatorial structure."* Knowing where quantum methods don't apply is more impressive than applying them everywhere indiscriminately.

### H.5 Validation Checklist for Module H

- [ ] Continuous solve (H.2) and binary-QUBO solve (H.3) tested independently — both produce weights summing to ~1.
- [ ] Both roughly track CRLB priors (Module E) in simple scenarios; investigate large unexplained deviations before trusting either.
- [ ] Binary version matches a brute-force check (discretize at 1/16 resolution, exhaustively search the simplex-constrained grid).
- [ ] J'(w) at the QIEO solution ≥ J'(w) at the classical-optimizer solution (Baseline 4/5) — or, if not, you have an honest result with analysis ready (Module N).

---

## 12. MODULE I — BQPhy / QuantumNow SDK Integration

### I.1 Confirmed Facts (checked directly against BQP's public materials)

- `bqphy.login()` and `bqphy.solve()` are the real, documented top-level entry points (BQP's own 3-step onboarding: Install → Authenticate via `bqphy.login()` → Solve via `bqphy.solve()`).
- The platform genuinely supports both **Continuous** and **Binary** optimization problem types (confirmed via BQP's MATLAB toolbox listing, same backend).
- Full API reference lives at a login-gated dev docs portal — your team needs SDK access to see exact class/parameter names. Everything else below this line is the **generic pattern**, consistent with what's publicly confirmed, but not a substitute for reading the real docs on Day 1.

### I.2 Step-by-Step Integration Protocol

**Step 1 — Get SDK access confirmed on Day 1, not Day 10.** This is the single highest-risk item on your schedule. Email organizers immediately asking whether pre-registration provides early BQPhy/QuantumNow access for prototyping.

**Step 2 — Run the SDK's own example notebook unmodified first.** Confirms environment, credentials, network access all work before you introduce your own bugs.

**Step 3 — Submit Module G's feature-selection QUBO** (smaller, fully brute-force-validated) as your first real test. Confirm it matches brute force.

**Step 4 — Submit Module H's fusion-weight problem** — try the continuous path (H.2) first; have the binary path (H.3) ready to swap in immediately if the continuous API doesn't behave as documented.

**Step 5 — Log everything**: wall-clock solve time, variable count, reads/iterations, returned objective value, for every solver call. This is raw data for Module J regardless of pipeline completion status elsewhere.

### I.3 Backup If SDK Access Is Delayed

Use `dimod`'s simulated annealing sampler as a drop-in placeholder with an identical interface contract, so swapping in the real solver later is a one-line change:

```python
# pip install dimod neal
import dimod, neal

def solve_qubo_classical_backup(Q):
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=1000)
    best = response.first
    return best.sample, best.energy
```

This lets G, H, J, K, L, M proceed in parallel regardless of SDK access timing.

### I.4 Validation Checklist for Module I

- [ ] SDK example notebook runs successfully.
- [ ] Module G QUBO submitted, result matches brute force.
- [ ] Module H submitted via both H.2 (continuous) and H.3 (binary) paths — at least one confirmed working before Day 10.
- [ ] Every solver call's timing/metadata logged to CSV for Module J.

---

## 13. MODULE J — Scaling Experiment (Your Strongest Slide, Done Honestly)

### J.1 Why This Module Is the Centerpiece

At your hackathon's actual problem size (3 sensors, 12–20 binary/continuous variables), classical methods will very likely be *faster in wall-clock time* — say this upfront. The defensible quantum-advantage argument is about how solve time and solution quality scale as the problem grows, not who wins at toy scale. **Per the audit (Section 2.5): do not cite BQP's own published 3.9x/20x/25x multipliers as evidence for your system.** Generate your own numbers.

### J.2 Protocol

Sweep 3→6→9→12 simulated sensor modalities (synthetic additional feature channels with matching statistical structure is standard practice for a scaling study — you don't need new physics for this). For each size, run: brute force (until infeasible, then stop and extrapolate), grid search, classical SLSQP (Module F.5), differential evolution (Module F.6), and QIEO (or its `neal` backup).

```python
import time

def run_scaling_experiment(problem_sizes, qubo_builder_fn, solver_fns: dict):
    results = []
    for n in problem_sizes:
        Q = qubo_builder_fn(n)
        for name, fn in solver_fns.items():
            start = time.time()
            solution, energy = fn(Q)
            results.append({"n_variables": n, "solver": name,
                             "solve_time_s": time.time() - start,
                             "objective_value": energy})
    return results
```

**Plot solve time (log y-axis) vs. problem size** AND **objective value achieved (normalized to best-known) vs. problem size**, one line per solver, side by side — speed without quality is a misleading metric, and showing you anticipated that is itself a credibility signal.

### J.3 Validation Checklist for Module J

- [ ] At smallest problem size, all solvers agree (validates correctness before trusting the trend).
- [ ] Both solve-time and solution-quality plots generated together.
- [ ] You have a prepared, honest explanation for whatever shape the curves actually take.
- [ ] Vendor benchmark numbers, if mentioned at all, are clearly attributed as motivating context, never as your own result.

---

## 14. MODULE K — Statistical Reliability Protocol

A single run with a single seed proves nothing to a PhD-level audience.

```python
from sklearn.model_selection import StratifiedKFold

def run_kfold_evaluation(X, y, pipeline_fn, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return [pipeline_fn(X[tr], y[tr], X[te], y[te]) for tr, te in skf.split(X, y)]

def run_multi_seed_experiment(seeds, full_pipeline_fn):
    all_metrics = []
    for s in seeds:
        np.random.seed(s)
        all_metrics.append(full_pipeline_fn())
    arr = np.array(all_metrics)
    return arr.mean(axis=0), arr.std(axis=0)

def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=1000, ci=95):
    from sklearn.metrics import roc_auc_score
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    lo, hi = np.percentile(aucs, (100-ci)/2), np.percentile(aucs, 100-(100-ci)/2)
    return np.mean(aucs), lo, hi
```

**Reporting standard:** every headline number = mean ± std (across seeds) or point estimate [95% CI] (bootstrapped AUC) — never a bare single number.

### K.1 Validation Checklist

- [ ] 5-fold CV implemented and run on the full pipeline.
- [ ] Minimum 5 random seeds used for scenario generation.
- [ ] Bootstrap CI computed for AUC on your headline comparison.
- [ ] Every number in final slides carries an uncertainty estimate.

---

## 15. MODULE L — Sensor Contribution Analysis

Three independent methods, used together — agreement is more convincing than any single method; disagreement is itself diagnostic.

```python
def ablation_study(weights_dict, sensor_scores, labels):
    from sklearn.metrics import roc_auc_score
    full_fused = sum(w*sensor_scores[s] for s, w in weights_dict.items())
    full_auc = roc_auc_score(labels, full_fused)
    results = {}
    for drop in weights_dict:
        remaining = {s: w for s, w in weights_dict.items() if s != drop}
        total = sum(remaining.values())
        renorm = {s: w/total for s, w in remaining.items()}
        ablated = sum(w*sensor_scores[s] for s, w in renorm.items())
        results[drop] = full_auc - roc_auc_score(labels, ablated)
    return results

import shap
def compute_shap_contributions(fused_model, X_features):
    explainer = shap.Explainer(fused_model, X_features)
    return explainer(X_features)
```

Method 1 (raw fusion weight magnitudes) is trivial but legitimate by construction — report as a bar chart.

### L.1 Validation Checklist

- [ ] All three methods (weight magnitudes, ablation, SHAP) computed on at least one representative scenario.
- [ ] Results compared side by side.
- [ ] Disagreements investigated before presenting, not just whichever supports the cleanest story.

---

## 16. MODULE M — Stretch Goal: Robustness Under Degradation

Build only after A–L are solid. A broken stretch-goal demo on stage is worse than no stretch goal.

```python
def inject_radar_jamming(range_doppler_map, jam_power_db):
    jam_noise = np.random.normal(0, 10**(jam_power_db/20), range_doppler_map.shape)
    return range_doppler_map + jam_noise

def degrade_thermal_sensor(thermal_frame, netd_multiplier):
    return thermal_frame + np.random.normal(0, netd_multiplier, thermal_frame.shape)

def elevate_ambient_noise(NL_dB, masking_increase_db):
    return NL_dB + masking_increase_db

def compare_adaptive_vs_static(degraded_scenario, static_weights, qieo_solver_fn):
    static_fused = fuse_with_weights(degraded_scenario, static_weights)
    adapted_weights = qieo_solver_fn(degraded_scenario)
    adaptive_fused = fuse_with_weights(degraded_scenario, adapted_weights)
    return static_fused, adaptive_fused
```

Plot detection-rate retention (% of clean-scenario performance) vs. degradation severity for adaptive vs. static. Expected, defensible story: static degrades faster, because re-optimization can shift trust away from the compromised sensor while static cannot. **Per Section 3.4, layer in a live-recomputed contribution-analysis figure here for an extra differentiator.**

### M.1 Validation Checklist

- [ ] At least 3 degradation scenarios implemented.
- [ ] Adaptive vs. static comparison run and plotted for each.
- [ ] Results reported honestly even if adaptive doesn't clearly win in some scenario.

---

## 17. MODULE N — Backup & Contingency Plans

| Risk | Likelihood | Backup Plan |
|---|---|---|
| QuantumNow/BQPhy access delayed | Medium | `neal`/`dimod` drop-in (Module I.3). Swap solver call only when SDK access arrives. |
| BQPhy continuous mode (H.2) doesn't work as documented | Medium | Binary QUBO path (H.3) is your certain fallback — keep it working in parallel, not as an afterthought. |
| QIEO shows no advantage over classical at tested scale | Medium | Pivot headline claim to the scaling trend (Module J), not raw speed. An honest negative result with rigorous analysis scores well under Documentation and signals intellectual maturity. |
| Dataset too easy (baselines near-perfect) | High if not managed | Increase clutter power, reduce target RCS/SL/contrast, narrow the SNR window until baseline detection sits ~60–75%. Do this in Module A–C validation, not late. |
| BQPhy SDK API differs from the generic pattern in this manual | Medium | Budget Day 1–2 for reading real docs and adapting — the math (Modules G, H) doesn't change, only the solver call interface does. |
| Dinkelbach-MIQ (G.2) doesn't converge in time | Low (since it's marked stretch) | Fall back to MID (G.1) — already your default and fully validated. |
| Running low on time before the PPT round | Medium | Priority order: A–C (real data) → F (credible baselines) → G–H (the actual quantum formulation) → J (scaling slide). L and M can be partially projected as "in progress" if clearly labeled. |
| Hardware/SDK rate limits or downtime during the event | Medium | Validate the full pipeline against `neal` before the event starts — a live outage doesn't stop your demo; show backup-solver results live and note QIEO results were pre-validated. |

---

## 18. MODULE O — Common Pitfalls and How to Catch Them

1. **Confusing σ (RCS) with σ_SB (Stefan-Boltzmann) or σ1²/σ0² (class variances) in code.** Use distinct variable names throughout: `rcs`, `stefan_boltzmann`, `class_var_1`, `class_var_0`.
2. **Forgetting xᵢ² = xᵢ for binary variables** when expanding quadratic penalty terms (Module G). Double-check every expansion against this identity.
3. **Testing the QUBO solver only on cases with tied optimal solutions** — hides formulation bugs. Always validate against a brute-force case with a confirmed unique optimum first.
4. **Reporting single-run numbers without uncertainty.** Module K is non-negotiable for credibility.
5. **Building Module M before A–L are validated.** Foundation first.
6. **Quietly under-tuned classical baselines** (default sklearn params, single-seed classical optimizer). Run with ≥10 random restarts — anything less makes "quantum beats classical" scientifically weak.
7. **Silently overwriting an array with a scalar** (the 1/f noise bug in Section 5/B.2) — always sanity-plot generated signals before trusting them downstream.
8. **Applying a CFAR threshold formula derived under one clutter distribution (Rayleigh) to a different one (Weibull) without re-deriving or empirically recalibrating it** (Section 4/A.3) — this is exactly the kind of error a PhD judge probes for.
9. **Citing a vendor's general marketing benchmark as if it's your own measured result** (Section 2.5) — produce your own numbers in Module J, every time.
10. **Trusting a fractional-programming convergence guarantee (Dinkelbach) when the inner solver is heuristic** (Section 2.2) — if you build the MIQ stretch, verify convergence empirically on your own data, don't cite the theorem as-is.

---

## APPENDIX A — Symbol Glossary

| Symbol | Module | Meaning |
|---|---|---|
| Pt, G, λ, R, σ (RCS), k, T0, B, F, L | A | Radar range equation parameters |
| h, c, σ_SB, ε, T | B | Thermal radiation parameters |
| SL, TL, NL, DI, DT | C | Sonar equation parameters |
| Relᵢ, Redᵢⱼ, α, β, γ, k (cardinality), λ_k (Dinkelbach) | G | Feature selection QUBO parameters |
| μ_c, σ_c², S_b, S_w, λ (Lagrangian/Rayleigh), w, x_{i,p} | H | Fusion weight optimization parameters |
| I(θ), CRLB | E | Classical Fisher Information / Cramér–Rao bound |
| Q | G, H | The QUBO matrix being minimized |

## APPENDIX B — Starting Hyperparameter Table (Copy-Paste Defaults)

```
Radar:      Pt=10000W, G=30dBi, f=10GHz, B=1MHz, F=4.77dB, L=6dB
            sigma_conventional=10 m^2, sigma_stealth=0.01-0.1 m^2
            CFAR (Rayleigh default): Pfa=1e-4, N_ref_cells=16
            CFAR (Weibull stretch): shape=1.5, Monte-Carlo calibrated, n_trials=200000

Thermal:    T_bg=290K, eps_bg=0.95
            T_conventional=400K, eps_conventional=0.9
            T_stealth=310K, eps_stealth=0.3
            NETD=0.05K (50mK), 1/f alpha=1.0-1.2

Acoustic:   SL_conventional=160dB, SL_stealth=110dB
            sea_state=3 (Knudsen), tonal freqs=[50, 120, 240] Hz
            absorption_coef=0.005 dB/m, LOFAR: fs=2000Hz, nperseg=1024, noverlap=512

QUBO #1 (MID default): alpha=1.0, beta=1.0, gamma=2.0, k_target=6-8 (of 12-16 features)
QUBO #1 (MIQ stretch):  damping=0.5, max_iter=15, tol=1e-4

Fusion weights: lambda=0.5 (Lagrangian); binary fallback: bits_per_weight=4, simplex_penalty=5.0
Classical baselines: n_restarts=10 (SLSQP), seed=42 (differential evolution)

Reliability: n_folds=5, n_seeds>=5, n_bootstrap=1000, CI=95%
```

---

**End of manual.** Build module by module, validate at every checklist, keep the binary-QUBO fallback for fusion weights working in parallel with the continuous path at all times, and never present a number without its uncertainty. Section 2 (the audit) is itself worth a slide — showing you can correctly evaluate a third party's technical claims, accept the good ones, and reject the overstated ones, is exactly the kind of judgment PhD judges are screening for.
