# Revising rumour decisions under late contradictory evidence — Phases 3–7

**What the numbers are.** All numbers are pooled over the 9 PHEME leave-one-event-out (LOEO) test folds under the **clean nested protocol**: every selection is made on validation events only.

**Seeds.** Results are the **mean ± std over 5 seeds** (7, 13, 21, 42, 99), unless marked "committed". *Committed* is the single seed-7 run this report was first written from (`artifacts/lifecycle/`). The 5-seed runs live in `artifacts/generalization/rq/baseline/seed*/` and `artifacts/generalization/baseline/seed*/`.

**Verdicts.** Every claim below matches its verdict in **`docs/claims_survival.md`** (claim IDs in brackets). That file also defines the noise floor and the CIs:
- **Noise floor:** same-seed run-to-run noise.
- **CIs:** crossed bootstrap over seeds × held-out events.
- **"Established":** the CI excludes 0 and the effect is larger than the noise.

## 1. Problem statement (draft)

Rumour verification systems are usually scored on one prediction made after a thread has finished. Real systems have to act while the thread is still growing. They may commit early, and they must decide whether to revise when later replies contradict that commitment. We study the **decision lifecycle**:

- **Abstain** (WAIT) while evidence is insufficient.
- **Commit** provisionally, then confirm.
- **Invalidate (REOPEN)** a commitment when *meaningful* contradictory evidence arrives.
- **Re-verify** and commit again.

We evaluate the whole decision trajectory, not just final accuracy. Our question is whether gating revision on evidence conflict produces better trajectories than three alternatives:
- never revising;
- revising whenever the classifier's prediction changes;
- deciding only at the end.

## 2. Contribution (clean protocol, 5 seeds; decision timing first, revision as a safety layer)

**What we study.** We ask two questions:
- when an early rumour-verification system should stop waiting and commit;
- whether a revision layer can safely withdraw commitments when contradicting evidence arrives late.

The setting is PHEME, with leave-one-event-out evaluation and causal, leakage-tested snapshots.

**Main finding: later propagation adds little on PHEME.**
- Held-out verifier accuracy rises only from 0.593 at 10% of the thread to 0.603 at 100%. Macro-F1 is flat (0.369 → 0.369) [V6].
- Committing every 4-class decision at 10% costs **≈ 1 pp** accuracy relative to waiting for the full thread: −0.95 ± 0.17 pp, negative in 5/5 seeds. Across held-out events the CI reaches 0 ([−1.55, +0.06] pp) [T1].
- An untuned "always commit at 10%" policy ties four tuned commit policies on earliness-weighted utilities [T2].
- A calibration-aware threshold commits on 90% ± 3% of threads within the first fifth of the conversation (median ≈ 3 min) [T4].
- That threshold **costs ≈ 0.5 pp** accuracy relative to waiting: −0.53 ± 0.13 pp, CI [−0.93, −0.08] pp. **The earlier "no detectable loss" does not survive** [T5].

**These are relative claims.**
- The verifier is below majority-class accuracy in 6–7 of 9 held-out events in every seed [V5].
- It beats the majority baseline clearly only on macro-F1: 0.369 vs 0.193, Δ CI [+0.114, +0.204] [V4].

**Revision as a safety layer.** We tested a conflict-gated revision layer (semantic strength vs volume, with near-duplicate collapsing):
- It does not measurably change final accuracy: −0.16 ± 0.14 pp, CI [−0.49, +0.02] pp [H1]. Harmed threads slightly outnumber helped ones in every seed [H2].
- It costs a little timing utility: w0.5 −0.0047 ± 0.0014, CI [−0.0123, −0.0003] [H3].
- **It is not established to detect injected late contradictions better than revising on prediction changes** [D1]:
  - reopen F1 is 0.20 ± 0.08 vs 0.10 ± 0.02, Δ +0.10 ± 0.09, with per-seed Δ ranging from −0.02 to +0.24;
  - the CI is [−0.08, +0.28];
  - same-seed run-to-run noise on this Δ is 0.11, larger than the effect.

The single-run result (+0.14 [+0.10, +0.17]) was one draw from this distribution.

**Why the safety layer is fragile.**
- Recall is low: 0.13 ± 0.06 [D3].
- Whatever detection there is comes from the folds where validation picked the less conservative gate: 94% ± 6% of correct reopens. *Which* folds those are changes with the seed (2–5 of 9) [D4].

**Not claimed:**
- absolute accuracy gains;
- that revision improves natural-thread accuracy;
- that the temporal verifier beats the existing GAT [V2, V8];
- that revised labels are correct;
- that conflict-gated revision beats prediction-change revision [D1];
- any veracity-only result (below majority-class accuracy; moved to limitations) [V10].

## 3. Method

### 3.1 Temporal verifier (Phase 3; TRIDENT removed, DGTR-style)
Code: `lifecycle/verifier.py`. It follows Wei et al. (2023), *DGTR: Dynamic Graph Transformer for Rumor Detection*: a shared structural encoder over a sequence of nested propagation sub-graphs, then a temporal transformer.

**Structural encoder.** A 2-layer GAT over frozen BERTweet node vectors plus prefix-local features:
- log minutes since the source post;
- causal depth;
- is-source flag;
- out-degree within the prefix.

It pools the source node, the mean and the max.

**Sub-graph sequence.** For an observed prefix, the sub-graphs are cut at **absolute deadlines**: 5 min, 15 min, 1 h, 4 h, 24 h, and everything observed. A 1-layer causal transformer runs over them.

**Deviation from DGTR, on purpose.** The model never sees the checkpoint index or fraction. "10% of the thread" implies the thread's final size, which correlates with the label, so it would leak the future.

**Training and calibration.** Class-weighted cross-entropy over all checkpoint prefixes, with the epoch chosen on the validation event. Temperature scaling (`decision/calibration.py`) is fitted on the validation event.

**Leakage check.** `tests/test_verifier.py` perturbs every post after the prefix (embedding, timestamp, parent) and asserts the prefix logits are unchanged. Parents that arrive after their child are not used.

**Baselines, all on the same prefixes** (`lifecycle/baselines.py`):
- `source_lr`: source tweet only;
- `prefix_lr`: mean BERTweet over the prefix;
- `gat_full`: your `models/gnn.py` GAT with BERTweet + 5 structural features, class weights and train-only normalisation, trained on full graphs.

### 3.2 Evidence store and conflict detector (Phase 4)
Code: `lifecycle/stance.py` and `lifecycle/evidence.py`.

**Stance model.**
- Supervision: RumourEval-2019 labels covering 4,709 PHEME replies (figshare 8845580, md5-verified).
- Model: logistic regression on word/char TF-IDF plus down-weighted BERTweet [reply, source, reply·source].
- Trained per fold **without** the held-out event's labels.
- Held-out macro-F1 is 0.36–0.65 depending on event (0.40–0.47 for most). Deny-F1 is only ≈0.1–0.2.

**Evidence store.**
- One item per reply.
- Near-duplicates (same normalised text with at least 3 tokens, or BERTweet cosine ≥ 0.95) form one cluster.
- Each cluster counts once, and only clusters *first seen after the commitment* are "new".

**Evidence model.** P(stance | label) is fitted on training-event threads. For a committed label L:

LLR(s, L) = log P(s | ¬L) − log P(s | L)

**Per-cluster conflict and support** use the positive and negative parts of the LLR.
- **strength** = the strongest single cluster's conflict, normalised to [0, 1].
- **volume** = the number of distinct clusters with normalised conflict ≥ 0.25.
- **net** = conflict minus β·support, counting only stance-bearing clusters. Neutral comments can therefore never accumulate.

**Meaningful contradiction** = either path:
- **Semantic path:** strength ≥ θ_strong, and net ≥ 0.
- **Volume path:** volume ≥ v_min, and net ≥ θ_net·κ, where κ = 1.5 for CONFIRMED (hysteresis).

### 3.3 Lifecycle (Phase 5)
Code: `lifecycle/state_machine.py`.

**WAIT → PROVISIONAL** when all three hold:
- max p ≥ τ_prov;
- evidence units ≥ min_units;
- the same label for 2 consecutive checkpoints.

**PROVISIONAL → CONFIRMED** when all four hold:
- p ≥ τ_conf;
- at least 3 evidence units;
- stable;
- conflict net below θ_net/2 (consistency check).

**REOPEN** fires on the gate. Re-verification then runs **immediately** on the same snapshot:

log q = log p_verifier + γ · Σ_clusters log P(stance | y)

The result is PROVISIONAL(argmax q) if q is confident, otherwise WAIT. A contradiction at the very last checkpoint therefore still gets a re-verified outcome.

**Tuning (validation event only).**
- **Stage A:** τ_prov, τ_conf and min_units, with no revision. These values are shared by baselines (b), (c) and (d), so the systems differ only in revision.
- **Stage B:** θ_net, θ_strong, v_min and γ.
- **Objective:** chance-corrected utility = mean over threads × checkpoints of +1 (committed and correct), −1/(C−1) (committed and wrong), 0 (WAIT). Uniform guessing scores 0.

**Edge-case tests** (`tests/test_lifecycle.py`, `tests/test_evidence.py`):

| Edge case | Test |
|---|---|
| 1. Very late contradiction after CONFIRMED | reopens and re-verifies at the last checkpoint; the locked-CONFIRMED ablation does not |
| 2. Repeated / near-duplicate contradictions | 4–8 copies count once and do not reopen; without dedup they do |
| 3. Insufficient evidence | a single-post thread and low-confidence threads stay WAIT |
| 4. Strong contradiction from very few posts | one strong reply reopens; two weak ones do not; support can outweigh it |
| Prediction flip without evidence | does not reopen under the conflict gate; does under gate (c) |

## 4. Experiments (Phase 6)

**Data.**
- 4-class: 6,424 threads (the single `unknown` thread dropped).
- Veracity (rumours only, 3-class): 2,402 threads.

**Split.** LOEO over 9 events. Inner validation = the next *rich* event in the sorted cycle (≥ 200 threads and ≥ 3 classes with ≥ 30 threads each). The old next-event rule made ebola-essien (14 threads, one class) the validation set for the charliehebdo fold.

**Seeds and noise.**
- Everything below is the mean ± std over 5 seeds of the clean protocol.
- `gat_full` previously ignored `--seed` (always seed 7); this is fixed.
- DGTR training is not bit-reproducible, because of multithreaded GAT scatter plus validation epoch selection. Rerunning *the same seed* moves pooled 4-class macro-F1 by 0.019 and accuracy by 2.4 pp.
- Differences smaller than that same-seed noise are treated as not established (`docs/claims_survival.md`).

### 4.1 Verifiers (calibrated, pooled; macro-F1 / accuracy / ECE; mean ± std over 5 seeds)

| task | model | 10% | 50% | 100% |
|---|---|---|---|---|
| 4-class | **dgtr (ours)** | 0.369±0.011 / 0.593±0.003 / 0.149±0.022 | 0.369±0.015 / 0.600±0.004 / 0.147±0.022 | **0.369±0.016** / 0.603±0.004 / 0.147±0.023 |
| 4-class | gat_full (existing GNN+BERTweet) | 0.345±0.006 / 0.543±0.015 / 0.114±0.034 | 0.344±0.006 / 0.546±0.012 / 0.115±0.033 | 0.341±0.006 / 0.548±0.010 / 0.115±0.030 |
| 4-class | source_lr | 0.321 / 0.540 / 0.124 | 0.321 / 0.540 / 0.124 | 0.321 / 0.540 / 0.124 |
| 4-class | prefix_lr | 0.311 / 0.541 / 0.114 | 0.314 / 0.544 / 0.106 | 0.313 / 0.546 / 0.101 |
| veracity | dgtr (ours) | 0.301±0.020 / 0.345±0.025 / 0.140±0.010 | 0.294±0.021 / 0.334±0.032 / 0.142±0.009 | 0.292±0.022 / 0.329±0.033 / 0.147±0.011 |
| veracity | gat_full | 0.293±0.020 / 0.303±0.021 / 0.102±0.021 | 0.294±0.020 / 0.302±0.022 / 0.101±0.022 | 0.286±0.023 / 0.294±0.024 / 0.107±0.024 |
| veracity | source_lr | 0.255 / 0.256 / 0.135 | 0.255 / 0.256 / 0.135 | 0.255 / 0.256 / 0.135 |
| veracity | prefix_lr | 0.276 / 0.280 / 0.128 | 0.288 / 0.291 / 0.112 | 0.279 / 0.282 / 0.119 |

The LR baselines are deterministic (lbfgs), so they have no seed variance.

**Δ macro-F1 @100%, dgtr − other** (5-seed mean ± std; 95% CI from a crossed bootstrap over seeds × held-out events):

| task | vs gat_full | vs source_lr | vs prefix_lr |
|---|---|---|---|
| 4-class | +0.027 ± 0.019 [−0.005, +0.047] (tie) [V2] | **+0.047 ± 0.016 [+0.013, +0.067]** [V3] | **+0.056 ± 0.016 [+0.023, +0.077]** [V3] |
| veracity | +0.005 ± 0.029 [−0.048, +0.055] (tie) [V8] | +0.037 ± 0.022 [−0.022, +0.097] | +0.013 ± 0.022 [−0.050, +0.057] |

The committed single run reported DGTR *significantly worse* than GAT on veracity (−0.030). That does not survive: across 5 seeds the two tie [V8].

**Veracity is not learnable across events with any of these models.**
- DGTR's pooled accuracy is 0.329 ± 0.033, about 1/3, i.e. chance level for 3 classes [V9].
- It is below the training-majority baseline (0.444) in 4–5 of 9 events in every seed [V10].
- Validation temperatures of 0.85–58 across models (4-class DGTR: 1.3–15.7) show that raw models are over-confident on unseen events [V7].

### 4.2 Lifecycle on natural trajectories (pooled test; mean ± std over 5 seeds)

| task | system | utility | final acc | final mF1 | final coverage | earliness | reopens / 100 | reopen P | reopen R | false-reopen | justified reopens corrected | flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-class | (a) final-snapshot only | 0.078±0.001 | 0.603±0.004 | 0.369±0.016 | 1.00 | 1.00 | 0 | – | 0 | 0 | – | 0 |
| 4-class | (b) early, no revision | 0.278±0.071 | 0.601±0.004 | 0.370±0.015 | 0.70±0.10 | 0.22 | 0 | – | 0 | 0 | – | 0 |
| 4-class | (c) revise on prediction change | 0.279±0.071 | 0.603±0.004 | 0.369±0.016 | 0.70±0.10 | 0.22 | 2.1±0.6 | 0.73±0.02 | 0.05±0.01 | 0.01±0.01 | 0.40±0.02 | 0.016 |
| 4-class | (d) full, conflict-gated | 0.278±0.070 | 0.599±0.005 | 0.368±0.015 | 0.70±0.10 | 0.22 | 10.3±3.9 | 0.63±0.07 | 0.19±0.07 | 0.08±0.04 | 0.07±0.07 | 0.014 |
| veracity | (a) | −0.001±0.008 | 0.329±0.033 | 0.292±0.022 | 1.00 | 1.00 | 0 | – | 0 | 0 | – | 0 |
| veracity | (b) | −0.019±0.021 | 0.330±0.033 | 0.293±0.024 | 0.60±0.14 | 0.22 | 0 | – | 0 | 0 | – | 0 |
| veracity | (c) | −0.020±0.020 | 0.329±0.033 | 0.292±0.022 | 0.59±0.13 | 0.22 | 2.7±0.8 | 0.64±0.09 | 0.04±0.01 | 0.05±0.02 | 0.31±0.08 | 0.017 |
| veracity | (d) | −0.020±0.020 | 0.329±0.032 | 0.291±0.024 | 0.59±0.13 | 0.22 | 8.5±3.5 | 0.81±0.03 | 0.14±0.05 | 0.08±0.03 | 0.14±0.06 | 0.047 |

Utility *levels* vary across seeds mainly because of charliehebdo, the largest test event (2,079 threads). Validation picks τ_prov = 0.8 for that fold in 2 of 5 seeds (7 and 13), and then (b)–(d) never commit on it.

**Paired differences, 4-class, d − b** (5-seed mean ± std; CI over seeds × events):
- final accuracy: −0.16 ± 0.14 pp [−0.49, +0.02] — not established [H1];
- utility w0: −0.0021 ± 0.0022 [−0.0071, +0.0003] — not established [H4];
- utility w0.5: −0.0047 ± 0.0014 [−0.0123, −0.0003] — established, small [H3].

**Other paired comparisons, 4-class.**
- d − c utility is −0.0016 ± 0.0007: negative in 5/5 seeds, but this is a seed-level interval only [H4].
- d − a utility is +0.199 ± 0.070. It comes from deciding early, not from revision [H5].

**Reading reopen precision against the base rate.**

| task | wrong commitments (base rate) | (d) reopen precision | lift | lift in seeds | verdict |
|---|---|---|---|---|---|
| 4-class | 0.39 ± 0.05 | 0.63 ± 0.07 | +0.24 ± 0.09 | 5/5 | a real lift [H6] |
| veracity | 0.70 ± 0.03 | 0.81 ± 0.03 | +0.12 ± 0.05 | 5/5 | a modest lift; the earlier "≈ random" does not survive [A11] |

**Why revision doesn't pay off.** Reopens find wrong commitments, but only 7% ± 7% of justified reopens on 4-class actually end on the gold label (committed run: 19%) [H7]. The re-verifier uses stance evidence with LLR ≤ 0.2–0.5 nats, which rarely gives it enough signal to pick the right label.

**Oracle stance** (human labels on annotated test threads; 4-class subset) [H8]:
- reopen recall goes from 0.22 to 0.24 (Δ +0.02 ± 0.07). That is within same-seed noise, so it is not established;
- precision goes from 0.84 to 0.79;
- utility Δ is 0.000.

A better stance model alone would not fix this.

### 4.3 Controlled late-evidence injection (mean ± std over 5 seeds)

Reopen rate per scenario. The first two scenarios *should* reopen; the last three should not.

| task | system | strong_single | distinct_volume | weak_duplicates | neutral | supporting | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 4-class | (c) prediction change | 0.05±0.01 | 0.07±0.01 | 0.07±0.01 | 0.07±0.01 | 0.05±0.01 | 0.38±0.03 | 0.06±0.01 | 0.10±0.02 |
| 4-class | **(d) full** | 0.13±0.07 | 0.12±0.06 | 0.06±0.03 | 0.07±0.03 | 0.07±0.03 | 0.56±0.04 | 0.13±0.06 | 0.20±0.08 |
| 4-class | − dedup | 0.13±0.06 | 0.12±0.06 | 0.11±0.06 | 0.08±0.04 | 0.08±0.03 | 0.50±0.05 | 0.13±0.06 | 0.20±0.08 |
| 4-class | volume path only | 0.01±0.01 | 0.03±0.03 | 0.01±0.01 | 0.02±0.02 | 0.02±0.02 | 0.43±0.08 | 0.02±0.02 | 0.04±0.04 |
| 4-class | CONFIRMED locked | 0.11±0.06 | 0.10±0.05 | 0.04±0.03 | 0.06±0.03 | 0.06±0.02 | 0.56±0.04 | 0.10±0.05 | 0.17±0.07 |
| veracity | (c) prediction change | 0.05±0.02 | 0.07±0.02 | 0.07±0.02 | 0.05±0.01 | 0.06±0.01 | 0.41±0.03 | 0.06±0.02 | 0.10±0.02 |
| veracity | **(d) full** | 0.14±0.08 | 0.10±0.06 | 0.04±0.02 | 0.06±0.03 | 0.06±0.03 | 0.60±0.04 | 0.12±0.07 | 0.20±0.10 |
| veracity | − dedup | 0.13±0.07 | 0.10±0.06 | 0.10±0.04 | 0.05±0.02 | 0.06±0.03 | 0.51±0.07 | 0.12±0.06 | 0.18±0.09 |

**(d) − (c) reopen F1** (crossed CI over seeds × events):

| task | committed | 5 seeds | per seed | CI | noise | verdict |
|---|---|---|---|---|---|---|
| 4-class | +0.135 | +0.10 ± 0.09 | −0.02 … +0.24 | [−0.08, +0.28] | 0.11 | not established [D1] |
| veracity | +0.138 | +0.10 ± 0.11 | −0.07 … +0.23 | [−0.06, +0.25] | 0.0001 | not established [D6] |

**Gate selection drives the spread.** Validation picks the "current" (less conservative) gate in 2–5 of 9 folds depending on the seed. Those folds give 94% ± 6% of (d)'s correct 4-class reopens [D4]. In the other folds the legacy gate with θ_net = 2 is almost inert: with CONFIRMED hysteresis it needs net conflict ≥ 3. §5 shows a trajectory where conflict strength reaches 0.99 and still never reopens.

### 4.4 Ablations and what each edge-case handler does (5 seeds)

| Handler | Effect | Verdict |
|---|---|---|
| **Dedup (edge case 2)** | Removing it raises duplicate-spam reopens: 4-class 5.5% → 11.0% (+5.5 pp [+1.9, +12.8]); veracity 3.8% → 9.5% (+5.7 pp [+2.2, +10.0]). | survives [A1, A10] |
| **Semantic-strength path (edge case 4)** | Does essentially all the work. Strength-only ≈ full (F1 Δ −0.015 [−0.067, +0.000]); volume-only recall is 0.02 ± 0.02. | survives [A2] |
| **Volume path** | Contributes nothing at the tuned thresholds (v_min = 5, θ_net = 2 in most folds). | survives [A2] |
| **Late reopen of CONFIRMED (edge case 1)** | Locking CONFIRMED lowers injection recall by 2.4 pp [−6.5, −0.1]. On natural 4-class data it raises reopen precision by +0.09 ± 0.03 (seed t-CI [+0.05, +0.13]). | survives [A3, A4] |
| **Hysteresis κ** | No measurable effect (F1 Δ +0.001, within noise), because κ only scales the rarely used volume path. | survives [A5] |
| **Minimum evidence for commitment (edge case 3)** | The ablation is **vacuous**: validation picks min_units = 1 in 8–9 of 9 folds. It is unit-tested but not empirically exercised. | survives [A6] |
| **Support offset β** | Removing it raises natural reopen recall (+0.08 ± 0.03) and false reopens (+0.04 ± 0.02); utility does not change. | survives [A7] |
| **Bayesian re-verifier vs verifier-only** | 4-class: corrected after a justified reopen 7% vs 2% (Δ +0.05 ± 0.06, seed t-CI [−0.03, +0.13]) — **not established**. Utility is the same. | doesn't survive [A8] |

**Figures** (`artifacts/lifecycle/<task>/figures/`; from the committed run):
- `trajectory_example_1.png`: a justified reopen that corrected the decision;
- `trajectory_example_2.png`: a false reopen;
- `trajectory_example_3.png`: (c) flip-flops on prediction changes while (d) holds;
- `coverage_accuracy.png`: coverage and accuracy per checkpoint.

5-seed error-analysis trajectories are in `artifacts/generalization/rq/baseline/seed7/4class/error_analysis.md`.

## 5. Where it fails (honest notes)

1. **The PHEME signal is the binding constraint.** Under LOEO, verifier accuracy barely improves as the thread grows (+0.95 pp from 10% to 100%; macro-F1 flat) [V6]. Even human stance is weak evidence of veracity. "Late contradictory evidence" therefore rarely exists in PHEME in a detectable form.
2. **Deny stance is poorly detected** (F1 ≈ 0.1–0.2) [S1]. The oracle experiment shows that better stance would not raise utility, and its recall gain is within noise [H8].
3. **Calibration is fold-unstable**, with validation temperatures from 0.85 to 58. Thresholds tuned on one validation event transfer poorly: τ_prov sits at the lowest grid value (0.4) in 8–9 of 9 folds [A9]. In 2 of 5 seeds the charliehebdo fold gets τ_prov = 0.8 and never commits.
4. **The verifier is not better than the existing GAT.** It ties on 4-class and on veracity [V2, V8].
5. **The injection test is synthetic.**
   - Injected replies come from other threads, so topic match is imperfect.
   - "Strong" posts were ranked by the fold's own stance model, which favours the detector on the strong-single scenario.
   - Even with that advantage, (d)'s edge over (c) is not established under seed noise [D1].
6. **Snapshot checkpoints are fractions of the final thread size** (the protocol from Phases 0–2). The model input itself is leak-free, but the evaluation grid still depends on final size.
7. **Single-run results on this pipeline are unreliable below ≈ 0.02 macro-F1 / ≈ 2 pp accuracy.** On RQ3 the threshold is about 0.1 F1. The same seed does not reproduce DGTR exactly (§4), and validation-selected gates and thresholds amplify small logit differences.
8. **Example failure, a wrong early commit** (baseline seed 7, prince-toronto, gold *false*):
   - The verifier says *non-rumour* with p = 0.88 from the first post and commits at 20%.
   - The replies push back: conflict strength reaches 0.99, and net conflict reaches 1.21 over 5 distinct clusters.
   - It never reopens, because validation chose the legacy gate (θ_net = 2.0, so a CONFIRMED label needs net ≥ 3.0).

## 6. Why cross-event generalization is hard

**1. The event is recoverable from the text, even after masking.** A linear probe (5-fold CV over threads; 9 events; chance ≈ 0.11) predicts the event from the source-tweet BERTweet vector with **0.99** accuracy.

We then re-embedded every post with an identical pipeline (unmasked reproduction cosine 1.00000) after masking:
- hashtags → `HASHTAG`;
- spaCy `en_core_web_lg` persons → `PERSON`;
- locations (GPE/LOC/FAC) → `LOCATION`;
- organisations → `ORGANIZATION`.

@mentions and URLs were already masked by the original normalisation. After masking the probe still reaches **0.82** (thread mean: 0.97 → 0.87). Names are not the main cue: topic words ("hostage", "crash", "cartoonists", "shooting") identify the event.

**2. Removing the name cues does not help.** Paired over 5 seeds × 9 held-out events:

| model | 4-class Δ macro-F1 | 4-class seeds / events improved | binary Δ macro-F1 | binary seeds / events improved |
|---|---|---|---|---|
| DGTR | −0.008 ± 0.020 | 2/5 seeds, 4/9 events | −0.004 ± 0.015 | 2/5 seeds, 7/9 events |
| static models | +0.009 to +0.019 | – | −0.011 to +0.009 | – |

The static-model changes are not consistent across held-out events (Wilcoxon p ≥ 0.16). Masking makes 4-class accuracy rise more across snapshots (+1.75 vs +0.95 pp), but only because it lowers early accuracy (0.575 vs 0.593 at 10%). The final accuracy is lower too.

**3. The same model is ≈ 0.16 macro-F1 better when the test event is seen in training.** This is the SAMGAT protocol comparison (stage 0 on branch `samgat-diagnosis`, commit 6acff67). It uses our GAT+BERTweet on the binary rumour / non-rumour task:

| split | macro-F1 | note |
|---|---|---|
| SAMGAT's random stratified 5-fold (events mixed across train and test) | **0.831** | 0.848 with SAMGAT's test-fold epoch selection; the paper reports 0.865 |
| LOEO | **0.673** pooled (fold-mean 0.547) | 5-seed rerun here: 0.677 ± 0.006; DGTR 0.723 ± 0.008 |

The gap is protocol, not architecture. A random split lets the classifier use exactly what the probe detects: event-specific vocabulary and each event's label mix. An unseen event offers neither.

**4. Label priors shift between events, and representation changes cannot fix that.** The non-rumour share per held-out event ranges from 0.00 (ebola-essien) and 0.02 (prince-toronto, almost all *false*) to 0.78 (charliehebdo). The training events are 55–65% non-rumour, so the verifier predicts that prior. On prince-toronto and ebola-essien, 4-class macro-F1 is 0.01–0.05 in every seed.

**5. A few events dominate pooled scores.**
- charliehebdo and sydneysiege are 51% of all threads, so pooled and per-event metrics can disagree.
- On binary, DGTR beats `source_lr` pooled in 5/5 seeds (+0.027) but in only 5/9 held-out events (p = 0.82).
- Masking binary DGTR helps 7/9 events but lowers pooled macro-F1, because the two largest events drop.

Full numbers: `docs/generalization_results.md`; probe: `artifacts/generalization/event_probe.json`.

## 7. Future work (not run, and why)

**End-to-end BERTweet fine-tuning, with class-balanced loss, with and without masking.**
- **Implemented:** `experiments/generalization/finetune.py`. It trains a causal prefix classifier on (source, observed replies) with Cui et al. class-balanced CE, selects the epoch on the validation event, and writes the same fold format as the verifiers. Smoke-tested on one fold.
- **Not run because it is infeasible on the available hardware.** On an Apple M2 (MPS, bf16) one step takes 4.6 s (batch 32, 128 tokens). That is ≈ 50 min per fold, so 5 seeds × 9 folds × 2 tasks × 2 text variants ≈ **150 GPU-hours**.
- **Expected cost on a single A100:** ≈ 10–15 h.

**Event-adversarial training (gradient reversal on an event classifier).**
- **Implemented:** `VerifierConfig.num_events` / `adv_lambda` and `run_verifiers.py --adv-lambda`. It uses a DANN-scheduled GRL on DGTR's final representation, over training events only. Unit tests show inference is unchanged and validation/test events never reach the head.
- **Not run because the pre-registered gate was "only if masking helps"** (`artifacts/generalization/DECISIONS.md`), and masking did not help DGTR on either task.
- **The case for running it anyway:** the probe shows the event is still 82% identifiable after masking, which is exactly the signal a GRL would target.

**Making single runs trustworthy.**
- Deterministic GAT aggregation, or reporting ≥ 5 seeds by default.
- RQ3 in particular needs this. Its same-seed noise (0.11 F1) is the size of the effect it claims.

## 8. Deviations from the plan, recorded

**Clean protocol (current).** Both selection leaks listed below are fixed by `--clean` in `experiments/run_lifecycle.py` and `experiments/run_injection.py`:
- **Stance features** are chosen per fold on the validation event (`lifecycle.stance.select_stance_config`).
- **The gate design** is a validation-selected hyperparameter (`STAGE_B_GRID_CLEAN`).

Outputs are under `artifacts/lifecycle/<task>/variants/clean/` and `artifacts/lifecycle/stance_clean/`. The items below describe the earlier, leaky protocol.


- **TRIDENT removed entirely** (no verifiable source). DGTR-style is the only temporal model.
- **Inner validation rule changed** to the next *rich* event (see §4).
- **Gate semantics changed after one smoke-test fold** on veracity/germanwings, before the full runs. The semantic path no longer requires net ≥ θ_net, because at PHEME's LLR scale a single post could never trigger it, which contradicted edge case 4.
- **Weak-evidence floor added.** Only stance-bearing clusters enter the net sums; neutral comments were masking strong denials. This was found by a unit test.
- **Stance features switched** from embeddings-only to the TF-IDF+embedding hybrid after comparing held-out macro-F1 on 4 events (0.37–0.40 → 0.37–0.45). That comparison used *test-event* stance annotations: it is selection on held-out labels, affecting the stance model only.
- **Gate-change sensitivity check** (`experiments/gate_check.py`, `artifacts/lifecycle/gate_check.md`; the legacy gate is rerun via `--legacy-gate`). RQ2 conclusions hold under both gates and without the inspected fold. The 4-class injection advantage exists only under the current gate (legacy F1 0.01 vs (c) 0.11). *(Single committed run; the legacy-gate variant was not rerun under 5 seeds [S5].)*

## 9. Reproduce

```bash
# RumourEval 2019 (stance supervision): figshare 8845580 -> data/external/rumoureval2019 (unzip both zips)
.venv/bin/pip install scikit-learn matplotlib
.venv/bin/python -m pytest tests/ -q                                       # 42 tests
.venv/bin/python -m experiments.run_verifiers --task 4class                # Phase 3 (~25 min)
.venv/bin/python -m experiments.run_verifiers --task veracity
.venv/bin/python -m experiments.run_lifecycle --task 4class [--only-event E] # Phases 4-5, per fold
.venv/bin/python -m experiments.run_lifecycle --task 4class --summarize
.venv/bin/python -m experiments.run_injection --task 4class                # stress test
.venv/bin/python -m experiments.plot_trajectories --task 4class
.venv/bin/python experiments/print_summary.py 4class
```

The committed run uses seed 7; the 5-seed reruns are below. Tuned thresholds per fold are in `artifacts/lifecycle/<task>/folds/<event>.json`.

### Reframed analysis (RQ1–RQ3)
```bash
.venv/bin/python -m experiments.run_lifecycle --task 4class --legacy-gate [--only-event E]   # gate-change check
.venv/bin/python -m experiments.run_lifecycle --task 4class --legacy-gate --summarize
.venv/bin/python -m experiments.run_injection --task 4class --legacy-gate
.venv/bin/python -m experiments.gate_check > artifacts/lifecycle/gate_check.md
.venv/bin/python -m experiments.run_commit_policies --task 4class                       # RQ1, validation cross-fit
.venv/bin/python -m experiments.run_lifecycle --task 4class --clean [--only-event E]        # clean protocol
.venv/bin/python -m experiments.run_lifecycle --task 4class --clean --summarize
.venv/bin/python -m experiments.run_injection --task 4class --clean
.venv/bin/python -m experiments.majority_baseline > artifacts/lifecycle/majority_baseline.md
.venv/bin/python -m experiments.rq_analysis                                              # CIs + RQ1 figures (incl. clean)
```

### 5-seed rerun, claims audit, generalization experiments
```bash
# multi-seed verifiers (seed passed to every model, incl. gat_full)
.venv/bin/python -m experiments.run_verifiers --task 4class --seed S --output artifacts/generalization/baseline/seedS
experiments/generalization/run_rq.sh baseline artifacts/generalization/baseline S                 # 4-class RQ1–RQ3 (clean)
TASK=veracity experiments/generalization/run_rq.sh baseline artifacts/generalization/baseline S    # veracity
.venv/bin/python -m experiments.generalization.claims_audit --tasks 4class,veracity              # docs/claims_survival.md numbers
.venv/bin/python -m experiments.generalization.doc_tables --task 4class                           # §4 tables
.venv/bin/python -m experiments.generalization.masking                                            # event-cue masking + re-embedding
.venv/bin/python -m experiments.generalization.event_probe
.venv/bin/python -m experiments.generalization.compare --exp artifacts/generalization/masked --task 4class --models dgtr,gat_full
```
