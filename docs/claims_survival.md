# Claims survival under 5 seeds

**Scope.** Every claim in `docs/lifecycle_report.md` and `docs/professor_summary.md` is checked against the baseline rerun with 5 seeds (7, 13, 21, 42, 99).
- The protocol is the same clean nested LOEO: commit policy, gate design, gate thresholds and stance features are all selected on validation events only.
- It uses the same code, plus the `gat_full` seed fix. Before the fix, `gat_full` always trained with seed 7.
- The original docs were written from one seed-7 run, called **committed** below.

**Sources.**
- `artifacts/generalization/claims_audit_<task>.json`, produced by `experiments/generalization/claims_audit.py`.
- 5-seed tables from `experiments/generalization/doc_tables.py`.
- Per-seed runs in `artifacts/generalization/rq/baseline/seed*/`.

## How to read the table

**Noise.** Noise = |committed − rerun with the same seed 7|: same code, same data, same seed. DGTR training is not bit-reproducible: multithreaded GAT scatter plus validation epoch selection. Same-seed pooled 4-class macro-F1 moves by 0.019 and accuracy by 2.4 pp. LR baselines and stance models are deterministic, so their noise is 0.

**CI.** The 95% CI comes from a **crossed bootstrap over seeds × held-out events**: seeds and events are resampled independently, and pairing is kept within thread and seed. Natural-trajectory ablation metrics have no per-thread decomposition, so they use a 5-seed t-interval, marked "seed t-CI".

**Established.** A difference is *established* only if its CI excludes 0 **and** |5-seed mean| > noise. Anything smaller than the same-seed noise is treated as not established.

**Verdicts.**
- **Survives:** a claimed effect is established, or a claimed tie / no-effect stays not established.
- **Weaker:** same direction, but the magnitude, the qualifier or the scope has to change.
- **Doesn't survive:** a claimed effect is not established, or an established result contradicts the claim.

**Source codes:** LR = lifecycle_report.md, PS = professor_summary.md.

## Verifier

| # | claim (source) | committed | 5 seeds: mean ± std | CI (seeds × events) | noise | verdict |
|---|---|---|---|---|---|---|
| V1 | 4-class DGTR pooled macro-F1 @100% (LR §4.1) | 0.358 | **0.369 ± 0.016** | – | 0.019 | survives (level) |
| V2 | DGTR **ties** existing GAT on 4-class (LR §2, §4.1, §5.4; PS limit 5) | +0.013 | +0.027 ± 0.019 | [−0.005, +0.047] | 0.019 | **survives**: difference not established |
| V3 | DGTR beats source_lr / prefix_lr on 4-class (LR §4.1) | +0.037 / +0.045 | +0.047 ± 0.016 / +0.056 ± 0.016 | [+0.013, +0.067] / [+0.023, +0.077] | 0.019 | **survives** |
| V4 | DGTR beats majority only on macro-F1 (LR §2; PS limit 1) | 0.358 vs 0.193 | 0.369 vs 0.193 (Δ +0.176 ± 0.016) | [+0.114, +0.204] | 0.019 | **survives** |
| V5 | DGTR below majority-class accuracy in 6/9 events; pooled 0.575 vs 0.626 (LR §2; PS limit 1) | 6/9; −5.1 pp | below in 6–7/9 in every seed; pooled 0.603 vs 0.626 (Δ −2.3 ± 0.4 pp) | Δ: [−6.9, +2.8] pp | 2.4 pp | **weaker**: the per-event statement survives; the pooled accuracy gap is within noise, so "pooled below majority" is not established |
| V6 | Verifier accuracy barely moves as the thread grows; 0.568 → 0.575 (LR §5.1; PS RQ1) | +0.7 pp | 0.593 → 0.603 (+0.95 ± 0.17 pp); macro-F1 flat (−0.001 ± 0.006) | – | 0.1 pp | **survives** |
| V7 | Raw models are heavily over-confident; validation T 1.6–58 (LR §4.1, §5.3) | 1.6–58 | T 0.85–58 across models/tasks; 4-class DGTR 1.3–15.7; DGTR ECE 0.147 ± 0.023 | – | – | **survives** |
| V8 | Veracity: DGTR significantly *worse* than GAT, −0.030 [−0.050, −0.009] (LR §4.1, §5.4; PS limit 5) | −0.030 | +0.005 ± 0.029 (0.292 vs 0.286) | [−0.048, +0.055] | 0.008 | **doesn't survive**: no difference is established in either direction; DGTR ties GAT on veracity too |
| V9 | Veracity is not learnable across events: pooled accuracy < 1/3 (LR §4.1) | 0.285 | 0.329 ± 0.033 (DGTR); GAT 0.294 ± 0.024 | – | 0.008 | **weaker**: DGTR is at ≈ 1/3 (chance-level for 3 classes), not clearly below it |
| V10 | Veracity below majority in 4/9 events and pooled 0.285 vs 0.444 (PS limit 2) | 4/9; −15.9 pp | below majority in 4–5/9 events in every seed; pooled 0.329 vs 0.444 (Δ −11.5 ± 3.3 pp, 5/5 seeds negative) | Δ: [−24.5, +3.7] pp | 0.8 pp | **weaker**: the per-event statement survives; the pooled gap is consistent across seeds but not established across held-out events. Veracity stays in limitations. |

## RQ1: decision timing (4-class)

| # | claim (source) | committed | 5 seeds: mean ± std | CI (seeds × events) | noise | verdict |
|---|---|---|---|---|---|---|
| T1 | Committing everything at 10% costs only 0.67 pp [0.14, 1.17] vs waiting (LR §2; PS RQ1) | −0.67 pp | **−0.95 ± 0.17 pp** (5/5 seeds negative) | [−1.55, +0.06] pp | 0.08 pp | **weaker**: the cost is ≈ 1 pp, not 0.67; it is consistent across seeds but not established across held-out events |
| T2 | An untuned "always at 10%" ties four tuned policies on earliness-weighted utilities (LR §2; PS RQ1) | gap w0.5 +0.0012, hold +0.0003 | gap w0.5 +0.0009 ± 0.0019, hold +0.0003 ± 0.0005 | – | 0.0008 / 0.0003 | **survives**: the gap is within noise |
| T3 | Only w0 ranks waiting first, by ≈ 0.01 (rq_results; LR §2) | +0.011 | best tuned − always-10% at w0: +0.009 ± 0.005 | – | 0.002 | survives |
| T4 | Calibration-aware threshold commits on 88% of threads by 20%, median ≈ 3 min (LR §2; PS RQ1) | 88%, 2.9 min | **90% ± 3%, 2.9 min** | – | 1.8 pp | **survives** |
| T5 | …with **no detectable loss** vs waiting, −0.03 pp [−0.42, +0.34] (LR §2; PS RQ1, contribution sentence) | −0.03 pp | **−0.53 ± 0.13 pp** (5/5 seeds negative) | **[−0.93, −0.08] pp** | 0.28 pp | **doesn't survive**: a ≈ 0.5 pp loss is established. Raw-confidence (−0.79 pp) and cost-sensitive (−0.73 pp) policies also lose. |

## RQ2: revision does no harm (4-class, (d) vs (b))

| # | claim (source) | committed | 5 seeds: mean ± std | CI (seeds × events) | noise | verdict |
|---|---|---|---|---|---|---|
| H1 | Revision does not change final accuracy, −0.12 pp [−0.47, +0.20] (LR §2; PS RQ2) | −0.12 pp | −0.16 ± 0.14 pp | [−0.49, +0.02] pp | 0.03 pp | **survives**: no change established |
| H2 | Harmed and helped threads roughly cancel, 65 vs 57 (PS RQ2) | 65 / 57 | 38 ± 31 / 28 ± 27; harmed > helped in 5/5 seeds | – | 8 / 10 threads | **weaker**: consistently slightly net-harmful, but the net effect (H1) is not established |
| H3 | Revision costs a little timing utility, w0.5 −0.0039 (LR §2; PS RQ2) | −0.0039 | −0.0047 ± 0.0014 | [−0.0123, −0.0003] | 0.0013 | **survives** |
| H4 | d − b utility at w0 n.s.; d − c n.s. (LR §4.2) | −0.0029 / −0.0009 | −0.0021 ± 0.0022 / −0.0016 ± 0.0007 | w0: [−0.0071, +0.0003]; d − c: seed t-CI [−0.0025, −0.0007] | 0.0000 / 0.0013 | d − b **survives**; d − c **weaker**: slightly negative in 5/5 seeds (seed-level only) |
| H5 | d − a utility +0.127 comes from deciding early, not from revision (LR §4.2) | +0.127 | +0.199 ± 0.070 | – | 0.0003 | **survives** (an earliness bonus; it varies with whether charliehebdo commits at all) |
| H6 | (d) reopen precision is a real lift over the base rate of wrong commitments, 0.66–0.69 vs 0.45 (LR §4.2) | 0.69 vs 0.45 | 0.63 ± 0.07 vs 0.39 ± 0.05; lift +0.24 ± 0.09 (5/5 > 0) | seed t-CI [+0.13, +0.34] | 0.02 | **survives** |
| H7 | Only 15% of justified reopens end on the gold label (LR §4.2) | 0.19 (clean) | 0.07 ± 0.07 | – | 0.004 | **survives**, and the effect is stronger: revision corrects even less |
| H8 | Oracle stance raises reopen recall (0.25 → 0.41); precision 0.90 → 0.84; utility unchanged, so a better stance model alone would not fix it (LR §4.2; PS limit 3) | clean: 0.17 → 0.18 | recall 0.22 → 0.24 (Δ +0.02 ± 0.07); precision 0.84 → 0.79; utility Δ 0.000 | – | Δ recall 0.06 | "raises recall" **doesn't survive** (within noise); "would not fix utility" **survives** |

## RQ3: injected late contradictions (4-class, (d) vs (c))

| # | claim (source) | committed | 5 seeds: mean ± std | CI (seeds × events) | noise | verdict |
|---|---|---|---|---|---|---|
| D1 | The conflict gate detects injected contradictions better than prediction-change revision: F1 0.24 vs 0.11, Δ +0.14 [+0.10, +0.17] (LR §2; PS RQ3, contribution sentence) | +0.135 | **+0.10 ± 0.09** (per seed −0.02 … +0.24) | **[−0.08, +0.28]** | **0.108** | **doesn't survive**: the same-seed noise is larger than the effect, and the CI includes 0 |
| D2 | (d) reopen F1 0.24 (LR §2; PS RQ3) | 0.236 | 0.20 ± 0.08 (0.10 – 0.32) | – | 0.087 | weaker (level is seed-fragile) |
| D3 | Recall is low, 0.15 (LR §2; PS RQ3) | 0.148 | 0.13 ± 0.06 | – | 0.076 | **survives** |
| D4 | The advantage comes from the 2/9 folds where validation picked the less conservative gate; elsewhere the gate is almost inert (LR §2; PS RQ3) | 2/9 folds; 98.7% of correct reopens | the current gate is picked in 2–5/9 folds (3.6 ± 1.1); those folds give 94% ± 6% of correct reopens | – | 2 folds | **weaker**: the mechanism survives; "2 of 9" is one draw of a seed-dependent selection |
| D5 | The full system does not reliably reopen on a strong single contradiction (rq_results) | 0.17 | 0.13 ± 0.06 | – | 0.07 | **survives** |
| D6 | Veracity: (d) vs (c) F1 0.24 vs 0.12; precision 0.73 vs 0.39 (LR §4.3) | F1 0.25 vs 0.12 (Δ +0.138); precision 0.67 vs 0.40 | F1 0.20 ± 0.10 vs 0.10 ± 0.02 (Δ +0.096 ± 0.108, per seed −0.07 … +0.23); precision 0.60 vs 0.41 | Δ F1: [−0.055, +0.253] | 0.0001 | **doesn't survive**: the CI includes 0 |

## Ablations (4-class)

| # | claim (source) | committed | 5 seeds: mean ± std | CI | noise | verdict |
|---|---|---|---|---|---|---|
| A1 | Removing dedup raises duplicate-spam reopens, 9% → 13% (LR §4.4) | +3.7 pp | 5.5% → 11.0% (+5.5 ± 4.7 pp) | [+1.9, +12.8] pp | 2.1 pp | **survives** |
| A2 | The semantic-strength path does essentially all the work; volume-only recall ≈ 0.01 (LR §4.4) | 0.005 | strength-only − full F1: −0.015 ± 0.020 [−0.067, +0.000]; volume-only recall 0.020 ± 0.020 | – | 0.03 | **survives** |
| A3 | Locking CONFIRMED lowers injection recall, 0.21 → 0.18 (LR §4.4) | −2.8 pp | −2.4 ± 1.5 pp | [−6.5, −0.1] pp | 0.6 pp | **survives** |
| A4 | Locking CONFIRMED raises natural reopen precision, 0.66 → 0.74 (LR §4.4) | +0.064 | +0.090 ± 0.030 | seed t-CI [+0.053, +0.127] | 0.009 | **survives** |
| A5 | Hysteresis has no measurable effect (LR §4.4) | +0.002 F1 | +0.0014 ± 0.0015 | [+0.0000, +0.0051] | 0.002 | **survives**: within noise |
| A6 | The min-evidence ablation is vacuous: min_units = 1 chosen in 8/9 folds (LR §4.4) | 8/9 | 8–9/9 (8.6) | – | 0 | **survives** |
| A7 | Removing the support offset raises reopen recall and false reopens; utility unchanged (LR §4.4) | recall +0.095, FRR +0.053 | recall +0.076 ± 0.034, FRR +0.040 ± 0.023, utility −0.0004 | seed t-CI [+0.034, +0.118] / [+0.011, +0.069] | 0.005 / 0.014 | **survives** |
| A8 | Only the Bayesian re-verifier can change the label after a reopen: 15% vs 3% corrected (LR §4.4) | 0.19 vs 0.04 | 0.07 ± 0.07 vs 0.02 ± 0.01 (Δ +0.05 ± 0.06) | seed t-CI [−0.025, +0.129] | 0.001 | **doesn't survive**: the difference is not established |
| A9 | τ_prov sits at the lowest grid value (0.4) in 8/9 folds (LR §5.3) | 8/9 | 8–9/9 (8.4) | – | 0 | **survives** |
| A10 | Veracity: removing dedup raises duplicate reopens, 3.8% → 10.8% (LR §4.3) | +4.4 pp | 3.8% → 9.5% (+5.7 ± 2.4 pp) | [+2.2, +10.0] pp | 3.3 pp | **survives** |
| A11 | Veracity: (d)'s reopen precision (0.75) is roughly what random reopening gives, base rate 0.74 (LR §4.2) | lift +0.03 | 0.81 ± 0.03 vs base 0.70 ± 0.03; lift +0.12 ± 0.05 (5/5 > 0) | seed t-CI [+0.05, +0.18] | 0.004 | **doesn't survive**: a modest lift is established at seed level |
| A12 | Veracity: revision does not change final accuracy, −0.04 pp (rq_results) | −0.04 pp | −0.09 ± 0.28 pp | [−0.49, +0.31] pp | 0.17 pp | **survives** |

## Seed-independent claims (re-checked, not re-estimated)

| # | claim (source) | why not seed-dependent | verdict |
|---|---|---|---|
| S1 | Stance model held-out macro-F1 0.36–0.65; deny F1 ≈ 0.1–0.2 (LR §3.2, §5.2) | the same cached, deterministic stance models are used in every seed | survives |
| S2 | Human stance shifts veracity odds by at most ≈ 1.7× per reply (PS limit 3) | a property of the annotations | survives |
| S3 | The injection test shows detection, not correctness of the revised label (PS limit 4) | a property of the design | survives; detection itself is now not established (D1) |
| S4 | Leak fixes shrank the detection advantage from +0.21 to +0.14 (PS limit 6) | historical, single run | superseded by D1 |
| S5 | The legacy-gate sensitivity check (LR §6) | the legacy-gate variant was not rerun with 5 seeds | not re-tested |

## Net effect on the headline story

- **Still supported.**
  - On PHEME, later propagation adds little (V6, T1–T4).
  - The verifier beats the LR baselines and the majority baseline on macro-F1 but not the existing GAT (V2–V4).
  - A revision layer does not measurably change final accuracy but costs a little timing utility (H1, H3).
  - The individual edge-case handlers behave as designed (A1–A7).
- **Withdrawn.**
  - "No measurable cost" for the calibration-aware commit rule (T5): it costs ≈ 0.5 pp.
  - "Conflict-gated revision detects late contradictions better than prediction-change revision" (D1): it is not established under seed noise.
  - The Bayesian re-verifier's correction advantage (A8).
  - On veracity:
    - "DGTR is worse than GAT" (V8): they tie;
    - the detection advantage (D6);
    - "reopen precision ≈ random" (A11): there is a small lift.
- **Unchanged in conclusion, weaker in wording.**
  - Veracity stays out of scope: below majority in 4–5/9 events, pooled −11.5 pp in every seed.
