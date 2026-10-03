# Revising rumour decisions under late contradictory evidence — Phases 3–7

All numbers are pooled over the 9 PHEME leave-one-event-out (LOEO) test folds, seed 7.
Sources: `artifacts/lifecycle/<task>/summary.json`, `injection_summary.json`, `folds/*.json`.
Tables are regenerated with `.venv/bin/python experiments/print_summary.py <task>`.

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

## 2. Contribution bullets (only what the results support)

1. **An evaluation protocol and open implementation of the rumour-decision lifecycle on PHEME.**
   - Causal snapshots at 10/20/30/50/75/100% of each thread.
   - Event-held-out evaluation.
   - Trajectory metrics: earliness, coverage and accuracy per checkpoint, reopen precision/recall, false-reopen rate, time-to-correct, flips, and chance-corrected utility.
   - A controlled late-evidence injection test, which supplies the "a reopen should happen here" ground truth that PHEME lacks.
2. **An evidence store and conflict detector.**
   - It separates *semantic strength* (one confident contradicting reply) from *volume* (many distinct weaker replies).
   - Near-duplicate replies collapse into one cluster.
   - Unit tests cover every required edge case (40 tests, all passing).
3. **Mechanism result (injection test, 4-class).** Conflict-gated reopening targets the right scenarios far better than revision on prediction change:
   - reopen precision 0.60 vs 0.36;
   - F1 0.31 vs 0.11.
   - Deduplication cuts reopens caused by duplicated weak contradictions from 13% to 9%.
   - The strength path does almost all of the work; the volume path is nearly inactive at the tuned thresholds.
4. **Negative empirical findings** that are themselves informative:
   - **Revision does not improve outcomes on natural trajectories.** On natural PHEME trajectories, conflict-gated revision does **not** improve utility or final accuracy over no revision (Δ utility −0.0009, 95% CI [−0.0028, +0.0010]).
   - **The cause is the data, not only the stance model.** Even *human* stance labels move veracity odds by at most ≈1.7× per reply (log-likelihood ratio ≤ 0.55 nats).
   - **Extra posts barely help any verifier.** Verifier accuracy is essentially flat from 10% to 100% of a thread.

What **not** to claim:
- that the full system beats the baselines on natural data;
- that the temporal verifier beats the existing GAT (it is a statistical tie on 4-class and worse on veracity);
- that this is a fact-checking system.

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

### 4.1 Verifiers (calibrated, pooled; macro-F1 / accuracy / ECE)

| task | model | 10% | 50% | 100% |
|---|---|---|---|---|
| 4-class | **dgtr (ours)** | 0.358 / 0.568 / 0.146 | 0.357 / 0.571 / 0.149 | 0.358 / 0.575 / 0.153 |
| 4-class | gat_full (existing GNN+BERTweet) | 0.349 / 0.560 / 0.130 | 0.347 / 0.561 / 0.129 | 0.345 / 0.561 / 0.126 |
| 4-class | source_lr | 0.321 / 0.540 / 0.124 | 0.321 / 0.540 / 0.124 | 0.321 / 0.540 / 0.124 |
| 4-class | prefix_lr | 0.311 / 0.541 / 0.114 | 0.314 / 0.544 / 0.106 | 0.313 / 0.546 / 0.101 |
| veracity | dgtr (ours) | 0.274 / 0.306 / 0.137 | 0.268 / 0.289 / 0.151 | 0.266 / 0.285 / 0.155 |
| veracity | **gat_full** | 0.307 / 0.315 / 0.088 | 0.303 / 0.310 / 0.090 | 0.296 / 0.301 / 0.095 |
| veracity | source_lr | 0.255 / 0.256 / 0.135 | 0.255 / 0.256 / 0.135 | 0.255 / 0.256 / 0.135 |
| veracity | prefix_lr | 0.276 / 0.280 / 0.128 | 0.288 / 0.291 / 0.112 | 0.279 / 0.282 / 0.119 |

**Bootstrap Δ macro-F1, dgtr − other (95% CI):**

| task | vs gat_full | vs source_lr | vs prefix_lr |
|---|---|---|---|
| 4-class, 100% | +0.013 [−0.000, +0.028] (tie) | +0.037 [+0.022, +0.050] | +0.045 [+0.031, +0.059] |
| veracity, 100% | **−0.030 [−0.050, −0.009] (worse)** | not significant | not significant |

**Veracity is not learnable across events with any of these models.** Pooled accuracy is below 1/3, and per-event DGTR accuracy ranges from 0.16 to 0.42 (4-class: 0.05 on prince-toronto to 0.70 on ferguson). Validation temperatures of 1.6–58 across models show that the raw models are heavily over-confident on unseen events.

### 4.2 Lifecycle on natural trajectories (pooled test)

| task | system | utility | final acc | final mF1 | final coverage | earliness | reopens / 100 | reopen P | reopen R | false-reopen | justified reopens corrected | flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-class | (a) final-snapshot only | 0.072 | 0.575 | 0.358 | 1.00 | 1.00 | 0 | – | 0 | 0 | – | 0 |
| 4-class | (b) early, no revision | **0.200** | 0.573 | 0.360 | 0.61 | 0.22 | 0 | – | 0 | 0 | – | 0 |
| 4-class | (c) revise on prediction change | **0.200** | 0.575 | 0.358 | 0.60 | 0.22 | 2.7 | 0.72 | 0.07 | 0.02 | 0.37 | 0.020 |
| 4-class | (d) full, conflict-gated | 0.199 | 0.572 | 0.361 | 0.60 | 0.22 | 12.5 | 0.66 | 0.23 | 0.12 | 0.15 | 0.029 |
| veracity | (a) | −0.012 | 0.285 | 0.266 | 1.00 | 1.00 | 0 | – | 0 | 0 | – | 0 |
| veracity | (b) | −0.052 | 0.281 | 0.260 | 0.59 | 0.21 | 0 | – | 0 | 0 | – | 0 |
| veracity | (c) | −0.050 | 0.285 | 0.266 | 0.59 | 0.21 | 2.5 | 0.70 | 0.04 | 0.05 | 0.54 | 0.020 |
| veracity | (d) | −0.053 | 0.281 | 0.254 | 0.58 | 0.21 | 6.8 | 0.75 | 0.11 | 0.10 | 0.20 | 0.037 |

**Paired bootstrap, 4-class, d − b:**
- utility −0.0009 [−0.0028, +0.0010];
- final accuracy −0.0011 [−0.0045, +0.0023].

d − c is likewise not significant. d − a: utility +0.127 [+0.118, +0.137], which comes from deciding early, not from revision.

**Reading reopen precision against the base rate.** About 45% of 4-class commitments, and about 74% of veracity commitments, are wrong.
- 4-class: (d)'s precision of 0.66 is a real lift.
- Veracity: 0.75 is roughly what reopening at random would give.

**Why revision doesn't pay off.** Reopens find wrong commitments, but only 15% of justified reopens actually end on the gold label. The re-verifier, which uses stance evidence with LLR ≤ 0.2–0.5 nats, rarely has enough signal to pick the right label.

**Oracle stance** (human labels on annotated test threads; 4-class subset):
- reopen recall rises from 0.25 to 0.41;
- precision falls from 0.90 to 0.84;
- utility Δ −0.001 [−0.015, +0.011].

A better stance model alone would not fix this.

### 4.3 Controlled late-evidence injection (4-class; veracity is similar)

Reopen rate per scenario. The first two scenarios *should* reopen; the last three should not.

| system | eligible | strong_single | distinct_volume | weak_duplicates | neutral | supporting | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| (b) no revision | 1164 | 0 | 0 | 0 | 0 | 0 | – | 0 | 0 |
| (c) prediction change | 1123 | 0.05 | 0.08 | 0.09 | 0.07 | 0.06 | 0.36 | 0.06 | 0.11 |
| **(d) full** | 1144 | **0.25** | **0.16** | 0.09 | 0.10 | 0.09 | **0.60** | **0.21** | **0.31** |
| − dedup | 1146 | 0.24 | 0.16 | 0.13 | 0.11 | 0.09 | 0.55 | 0.20 | 0.30 |
| strength path only | 1145 | 0.25 | 0.16 | 0.09 | 0.10 | 0.08 | 0.60 | 0.21 | 0.31 |
| volume path only | 1161 | 0.00 | 0.01 | 0.00 | 0.01 | 0.01 | 0.40 | 0.01 | 0.01 |
| CONFIRMED locked | 1144 | 0.22 | 0.14 | 0.08 | 0.09 | 0.08 | 0.60 | 0.18 | 0.27 |
| − hysteresis | 1144 | 0.25 | 0.16 | 0.09 | 0.10 | 0.09 | 0.60 | 0.21 | 0.31 |

Veracity, (d) vs (c): precision 0.73 vs 0.39, F1 0.24 vs 0.12. Dedup: 3.8% vs 10.8% on weak duplicates.

### 4.4 Ablations and what each edge-case handler does

| Handler | Effect | Measured where |
|---|---|---|
| **Dedup (edge case 2)** | Removing it raises duplicate-spam reopens from 9% to 13% (4-class) and from 4% to 11% (veracity). | injection test |
| **Semantic-strength path (edge case 4)** | Does essentially all the work. Removing it (volume only) drops reopen recall to about 0.01. | injection and natural data |
| **Volume path** | Contributes nothing at the tuned thresholds (v_min = 5, θ_net = 2 in most folds). | tuning output |
| **Late reopen of CONFIRMED (edge case 1)** | Locking CONFIRMED lowers injection recall 0.21 → 0.18. On natural 4-class data it raises reopen precision 0.66 → 0.74 at slightly lower recall. | injection and natural data |
| **Hysteresis κ** | No measurable effect, because κ only scales the rarely-used volume path. | — |
| **Minimum evidence for commitment (edge case 3)** | The ablation is **vacuous**. Validation picked min_units = 1 in 8 of 9 folds, so it is unit-tested but not empirically exercised. | — |
| **Support offset β** | Removing it raises reopen recall (0.23 → 0.32) and false reopens (0.12 → 0.17). Utility does not change. | natural data |
| **Bayesian re-verifier vs verifier-only** | Only the Bayesian version can change the label after a reopen (15% vs 3% corrected), but utility is the same. | natural data |

**Figures** (`artifacts/lifecycle/<task>/figures/`):
- `trajectory_example_1.png`: a justified reopen that corrected the decision;
- `trajectory_example_2.png`: a false reopen;
- `trajectory_example_3.png`: (c) flip-flops on prediction changes while (d) holds;
- `coverage_accuracy.png`: coverage and accuracy per checkpoint.

## 5. Where it fails (honest notes)

1. **The PHEME signal is the binding constraint.** Under LOEO, verifier accuracy does not improve as the thread grows. Even human stance is weak evidence of veracity. "Late contradictory evidence" therefore rarely exists in PHEME in a detectable form.
2. **Deny stance is poorly detected** (F1 ≈ 0.1–0.2). Fine-tuning BERTweet per fold is the obvious next step. The oracle experiment suggests it would raise reopen recall but not utility.
3. **Calibration is fold-unstable** (T = 2–58). Thresholds tuned on one validation event transfer poorly. In 8 of 9 folds (both tasks) τ_prov sits at the lowest grid value (0.4).
4. **The verifier is not better than the existing GAT.** It ties on 4-class and is worse on veracity.
5. **The injection test is synthetic.** Injected replies come from other threads, so topic match is imperfect. "Strong" posts were ranked by the fold's own stance model, which favours the detector on the strong-single scenario.
6. **Snapshot checkpoints are fractions of the final thread size** (the protocol from Phases 0–2). The model input itself is leak-free, but the evaluation grid still depends on final size.

## 6. Deviations from the plan, recorded

- **TRIDENT removed entirely** (no verifiable source). DGTR-style is the only temporal model.
- **Inner validation rule changed** to the next *rich* event (see §4).
- **Gate semantics changed after one smoke-test fold** on veracity/germanwings, before the full runs. The semantic path no longer requires net ≥ θ_net, because at PHEME's LLR scale a single post could never trigger it, which contradicted edge case 4.
- **Weak-evidence floor added.** Only stance-bearing clusters enter the net sums; neutral comments were masking strong denials. This was found by a unit test.
- **Stance features switched** from embeddings-only to the TF-IDF+embedding hybrid after comparing held-out macro-F1 on 4 events (0.37–0.40 → 0.37–0.45).

## 7. Reproduce

```bash
# RumourEval 2019 (stance supervision): figshare 8845580 -> data/external/rumoureval2019 (unzip both zips)
.venv/bin/pip install scikit-learn matplotlib
.venv/bin/python -m pytest tests/ -q                                       # 40 tests
.venv/bin/python -m experiments.run_verifiers --task 4class                # Phase 3 (~25 min)
.venv/bin/python -m experiments.run_verifiers --task veracity
.venv/bin/python -m experiments.run_lifecycle --task 4class [--only-event E] # Phases 4-5, per fold
.venv/bin/python -m experiments.run_lifecycle --task 4class --summarize
.venv/bin/python -m experiments.run_injection --task 4class                # stress test
.venv/bin/python -m experiments.plot_trajectories --task 4class
.venv/bin/python experiments/print_summary.py 4class
```

All runs use seed 7. Tuned thresholds per fold are in `artifacts/lifecycle/<task>/folds/<event>.json`.
