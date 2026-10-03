# Results under the clean protocol (RQ1–RQ3)

**Framing.** The core contribution is *decision timing*: when to stop waiting and commit. Revision is a *safety layer*.

**Supersedes** the earlier version of this file. "Previous" columns below are the earlier numbers, kept for comparison. They came from a protocol with two selection leaks, both now fixed.

**Sources:** `artifacts/lifecycle/rq_results.json`, `majority_baseline.md`, `gate_check.md`, `<task>/commit_policies/summary.json`, and `<task>/variants/clean/`.

**Statistics.** CIs are 95% paired bootstraps over threads (2,000 resamples). For RQ3 the resampling unit is the thread, keeping all its injection scenarios together.

## What changed: the clean protocol

**1. Reopen gate.**
- **Before:** the gate's semantic-path design was finalised after inspecting one test fold (veracity/germanwings).
- **Now:** the design is a hyperparameter. "Legacy" requires net ≥ θ_net on the semantic path too; "current" requires net ≥ 0. It is selected per LOEO fold **on the validation event only**, jointly with θ_net, θ_strong, v_min and γ, using the unchanged stage-B validation objective. Legacy is listed first, so ties keep the original design. The choice is then frozen and scored on test.
- **Selected:** 4-class legacy in 7/9 folds and current in 2/9 (prince-toronto, sydneysiege). Veracity current in 6/9 and legacy in 3/9.

**2. Stance-model features.**
- **Before:** chosen by comparing macro-F1 on four held-out events' annotations.
- **Now:** the same 7 candidate configurations (embedding C ∈ {0.005, 0.02, 0.1}; TF-IDF C ∈ {1, 3}; hybrid C ∈ {1, 3}) are scored per fold **on the validation event's annotations**, training on the remaining events. The winner is refit on all non-test events.
- **Selected:** hybrid in 7/9 folds and embedding-only (C = 0.005) in 2/9 (the folds whose validation event is ottawashooting).
- **Rerun:** all stance-dependent results (evidence model, conflict gate, re-verifier, RQ2, RQ3).

**3. What is unaffected.** RQ1 uses neither the stance model nor the gate, and its policy selection was already validation-only (2-way cross-fitted). Its numbers are unchanged.

## Applying your rule: does the verifier beat the majority-class baseline per fold?
The majority class comes from the training events. Details are in `artifacts/lifecycle/majority_baseline.md`.

| task | folds where held-out accuracy > majority | pooled acc: DGTR / existing GAT / majority | pooled macro-F1: DGTR / majority |
|---|---|---|---|
| veracity | 5 / 9 | 0.285 / 0.301 / **0.444** | 0.266 / 0.205 |
| 4-class | **3 / 9** | 0.575 / 0.561 / **0.626** | **0.358** / 0.193 |

- **Veracity fails, so it moves to limitations.** Its RQ results below are reported for completeness only.
- **4-class fails the same accuracy test, which you did not ask about but which matters.** Always predicting "non-rumour" is more accurate than the verifier in 6/9 folds. The verifier beats majority only on macro-F1 (0.358 vs 0.193).
- **Consequence:** every 4-class claim below is about **macro-F1 and relative timing** (early vs late decisions of the same verifier), not about absolute accuracy.

## RQ1 — Does later propagation add anything? (4-class; unchanged under the clean protocol)

**Reworded finding: on PHEME, later replies add little to the verifier, so commitment can happen very early.**

**The data.**
- Accuracy of the verifier's argmax is nearly flat across a thread: validation 0.585 at 10% → 0.593 at 100%; held-out 0.568 → 0.575.
- An **untuned "always commit at 10%"** policy ties the four tuned commit policies on both earliness-weighted utilities (validation, cross-fitted):

| policy (tuned on validation for each utility) | U w0 (no earliness reward) | U w0.5 | U hold |
|---|---|---|---|
| fixed checkpoint | **0.457** | 0.424 | 0.446 |
| raw-confidence threshold | 0.448 | 0.425 | **0.447** |
| calibration-aware threshold | 0.449 | 0.404 | 0.375 |
| cost-sensitive stopping | 0.452 | **0.426** | 0.446 |
| *always at 10% (untuned)* | 0.447 | 0.424 | **0.447** |
| *always at 100% (untuned)* | **0.457** | 0.229 | 0.076 |

**What waiting is worth (held-out, frozen; Δ accuracy vs final-snapshot decision on the same threads).**
- Always at 10%: **−0.67 pp [−1.17, −0.14]**. This is the total value of the remaining 90% of the replies.
- Calibration-aware threshold: −0.03 pp [−0.42, +0.34]. It commits on 88% of threads by 20% (median about 3 minutes after the source), abstains on 6%, and its validation risk–coverage curve coincides with final-snapshot thresholding.
- Cost-sensitive stopping: −0.50 pp [−0.92, −0.08].
- Raw confidence: −0.98 pp [−1.43, −0.53].

**Utility caveat.** Earliness-weighted utilities (w0.5, hold) cannot discriminate between policies here, because accuracy is flat. Only w0 ranks waiting first, and only by 0.01. The earlier "+0.127 over final-only" was an earliness bonus and is withdrawn.

**What this does and does not say.** RQ1 is a statement about *where the signal is*: in the source tweet and the first few replies. It does not show that early decisions are good in absolute terms; see the majority-baseline result above.

## RQ2 — Does revision harm ordinary decisions? (4-class; (b) no revision vs (d) conflict-gated)

| | previous protocol | **clean protocol** |
|---|---|---|
| threads with ≥ 1 reopen | 8.6% | 5.9% |
| reopens against wrong / correct commitments | 528 / 276 (precision 0.66) | 401 / 179 (precision 0.69); base rate of wrong commitments 0.45 |
| threads harmed / helped | 68 / 61 | 65 / 57 |
| **final accuracy d − b** | −0.11 pp [−0.47, +0.23] | **−0.12 pp [−0.47, +0.20]** |
| utility w0 d − b | −0.0028 [−0.0074, +0.0016] | −0.0029 [−0.0073, +0.0013] |
| utility w0.5 d − b | −0.0057 [−0.0094, −0.0021] | **−0.0039 [−0.0074, −0.0005]** |
| utility hold d − b | −0.0084 [−0.0118, −0.0052] | −0.0050 [−0.0081, −0.0021] |

**Finding.** Revision does not change final accuracy (CI within ±0.5 pp). There is a small, significant timing cost under earliness-weighted utilities, because a reopen withdraws a decision.

## RQ3 — Does revision catch late contradictions? (4-class injection test; (c) revise-on-prediction-change vs (d))

| | previous protocol | legacy gate forced | **clean protocol** |
|---|---|---|---|
| (d) reopen precision | 0.60 [0.57, 0.62] | 0.52 [0.40, 0.73] | **0.59 [0.56, 0.62]** |
| (d) reopen recall | 0.21 [0.19, 0.23] | 0.01 [0.00, 0.01] | **0.15 [0.13, 0.17]** |
| (d) reopen F1 | 0.31 [0.29, 0.33] | 0.01 [0.00, 0.02] | **0.24 [0.21, 0.26]** |
| (c) reopen F1 | 0.11 [0.09, 0.13] | 0.11 | 0.11 [0.09, 0.12] |
| **F1 d − c (paired)** | +0.21 [+0.18, +0.24] | −0.09 [−0.11, −0.07] | **+0.14 [+0.10, +0.17]** |
| (d) reopen rate, strong single / five distinct | 0.25 / 0.16 | 0.00 / 0.01 | 0.17 / 0.13 |
| (d) reopen rate, weak duplicates / neutral / supporting | 0.09 / 0.10 / 0.09 | ≈ 0 | 0.07 / 0.08 / 0.06 |
| duplicates with dedup removed | 0.13 | 0.02 | 0.11 |

**Where the clean effect comes from.** In the 7/9 folds where validation chose the legacy design, (d) is almost inert: 6 correct and 11 incorrect reopens in 3,080 injection cases. Prince-toronto and sydneysiege, where validation chose the current design, supply 332 of the 338 correct reopens. So the pooled advantage over (c) is real under a leak-free protocol, but on 4-class it rests on **2 of 9 folds**.

## Veracity (moved to limitations: fails the majority-baseline rule)
For completeness only:
- RQ2 clean: final acc d − b = −0.04 pp [−0.67, +0.54].
- RQ3 clean: (d) F1 0.25 [0.23, 0.27] vs (c) 0.12; Δ +0.14 [+0.11, +0.17]. Validation chose the current gate in 6/9 folds.
- RQ1: committing early is *more* accurate than waiting (+2.1 pp at 10%). With a below-majority verifier, this means "later replies make a weak model worse".

## Which claims survive

| claim | status |
|---|---|
| Later propagation adds little on PHEME; committing at 10% costs ≤ 0.7 pp vs waiting for the full thread (4-class) | **survives** (never touched by the leaks) |
| A calibration-aware rule commits within the first fifth of a thread with no detectable loss relative to waiting | **survives**, as a *relative* claim |
| Early decisions are accurate in absolute terms | **does not survive**: below majority-class accuracy in 6/9 folds; only macro-F1 beats majority |
| Revision does not harm final accuracy | **survives** (−0.12 pp [−0.47, +0.20]) |
| Revision has a small timing cost | **survives**, smaller (w0.5: −0.004) |
| Conflict-gated revision detects injected late contradictions better than prediction-change revision | **survives, weakened**: Δ F1 +0.14 [+0.10, +0.17] (was +0.21); on 4-class it comes from the 2/9 folds where validation picked the current gate |
| The full system reliably reopens on strong single contradictions | **does not survive**: 17% overall, and near 0% in the 7 legacy-gate folds |
| Any veracity (true/false/unverified) result | **moved to limitations** |
| The safety layer reaches high recall | **never claimed**: 0.15 under the clean protocol |
