# 📡 Deep Learning-Assisted Robust Channel Estimation over Phase-Drifting Backscatter Links

> Residual-gated CNN (RG-CNN) that suppresses the deterministic error floor of closed-form phase-aware estimators under large residual carrier-phase drift.

## 📌 About

In monostatic backscatter links, residual carrier-frequency offset (CFO) and reader-side synthesizer mismatch leave a slowly drifting carrier phase across the pilot block after coarse synchronization. This drift acts multiplicatively on the cascaded two-hop observation and biases channel estimation (CE).

Closed-form model-based estimators address the drift by **linearizing** the phase-slope-induced rotation across the pilot block (a first-order lifted model). This is accurate only while the accumulated drift stays within a bounded range. Beyond that range, the discarded curvature projects through the estimator as a bias that is **independent of noise**, so the CE error saturates at a deterministic floor regardless of SNR.

This project implements a **deep learning-aided phase-aware estimator** that keeps the closed-form lifted estimate as a model-based anchor and applies a **bounded, residual-gated correction**. A residual-gated convolutional neural network (RG-CNN) stays inactive while the linearized model holds and acts only under large drift — preserving model-based accuracy at low complexity while removing the floor.

## 🎯 Key Results

- **No error-floor saturation:** the lifted CE saturates near **−16.5 dB** by 30 dB SNR; the RG-CNN keeps improving to **−32.8 dB**, a **~16 dB** gain at high SNR
- **Span-induced floor suppression:** at a 120° pilot phase span, the noiseless floor is held at **−23.9 dB** — about **14 dB** below the lifted CE and **23 dB** below the conventional phase-stationary estimator
- **Extended admissible drift range:** at 30 dB the RG-CNN stays near −40 dB up to roughly **70°–80°** span, while the baselines depart below 40° — effectively extending the validity boundary Φ_th
- **Tail-failure reduction:** large-drift failure probability `P_fail = Pr[NMSE > −20 dB]` drops from **0.97** (lifted CE) / **1.00** (conventional) to **0.29** (RG-CNN)
- **Compact and low-complexity:** ~**1.86 × 10⁴** trainable parameters, confined to the gate network; the initialization and correction direction are model-based, so training converges within ~100 iterations

## 🔧 System Architecture

**Signal model (pilot block of length M):**

```
r[m] = h · c[m] · e^{ j φ m } + z[m],    m = 0, ..., M-1
```

where `h = h_tt · h_tr` is the composite forward × backscatter gain (quasi-static, single complex scalar over the block), `c[m]` are antipodal BPSK pilots, `φ` is the per-sample residual phase rate, and `z[m]` is additive noise. The drift is summarized by the **pilot phase span** `Φ = |φ|(M-1)`.

**Model-based anchor — closed-form lifted LS:**
Expanding `e^{jφm} ≈ 1 + jφm` and lifting the unknowns as `u₀ = h`, `u₁ = jφh` recasts the observation as linear in `[u₀, u₁]`. The closed-form LS solution returns the anchor `ĥ₀ = û₀` and `φ̂₀ = ℑ{û₁/û₀}`.

**RG-CNN refinement:**
1. Reconstruct the pilot from the anchor and form the **model residual** `d[m] = c*[m](r[m] − r̂₀[m])`, which vanishes when the first-order model is exact and grows with the discarded curvature.
2. Build a per-index feature map `F ∈ ℝ^{7×M}` from the matched-filtered observation, the model residual, the normalized index, and the local phase coordinate (real/imaginary split, amplitude-normalized).
3. A compact backbone of three 1-D conv layers (GELU, dilation in layers 2–3) plus global average/max pooling, concatenated with frame-level summaries `s`, feeds a small fully connected gate head.
4. A **damped Gauss–Newton step** `δ` on the *exact* model, clamped to a trust region (`|δ_ℜ|, |δ_ℑ| ≤ κ_h`, `|δ_φ| ≤ κ_φ`), gives the model-based correction direction.
5. The gate `β = β_max · σ(f_Ω(F, s)) ∈ [0, β_max]` scales the correction: refined estimate `ψ̂ = ψ̂₀ + β·δ`.

Both `δ` and `β` are driven by the same residual `r − r̂₀`. While `Φ ≤ Φ_th` the residual is noise-dominated, the gate stays near zero, and `ψ̂ ≈ ψ̂₀` (closed-form accuracy preserved). As `Φ` grows and the residual exposes the curvature, the gate opens and the exact-model correction suppresses the floor.

**Training:**
Offline on synthetic pilots from the exact model. `|h|` log-uniform, phase uniform, SNR swept over a wide range, and `Φ` sampled across both the model-valid and large-drift regimes. The objective combines a channel-NMSE term with auxiliary phase, reconstruction, and correction-magnitude terms (channel NMSE kept dominant); optimized with AdamW under gradient-norm clipping, the gate initialized to a small value so optimization starts from the model-based estimate.

## 🛠️ Built With

`Python`

## 📄 Publication

**H. Ryu** and S. Kim, "Deep Learning-Aided Phase-Aware Channel Estimation for Error-Floor Suppression over Phase-Drifting Backscatter Links," *International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS)*, 2026.
