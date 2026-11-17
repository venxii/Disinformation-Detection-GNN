# Cross-event generalization (branch `generalization`, off `clean-protocol`)

**Protocol.** PHEME, leave-one-event-out (9 folds). The inner validation event is the next *rich* event, as in `lifecycle.data.loeo_folds`. Every choice was made on validation events only: epochs, temperatures, commit thresholds, gate design and stance features. Test events are scored once with frozen choices. Five seeds (7, 13, 21, 42, 99) everywhere. Comparisons are **paired** by (seed, held-out event).

**Statistics.** "Seeds Δ>0" counts seeds whose pooled LOEO macro-F1 improved. "Folds Δ>0" counts held-out events whose seed-averaged Δ is positive. The Wilcoxon test runs over those 9 event-level Δs, because the event is the unit of generalization.

**Decision rules.** These were written down before the full step-2 results were seen; see `artifacts/generalization/DECISIONS.md`.

**Code.** `experiments/generalization/`:

| file | what it does |
|---|---|
| `multiseed_summary.py` | step 1 summary |
| `masking.py` | masking and re-embedding |
| `event_probe.py` | event-identifiability probe |
| `compare.py` | paired comparison |
| `finetune.py` | step 3 (implemented; not run at scale) |
| `snapshot_curves.py` | accuracy across snapshots |
| `run_rq.sh` | RQ pipeline per seed |
| `rq_summary.py` | RQ summary across seeds |
| `error_analysis.py` | step 6 |
| `claims_audit.py` | claims audit: same-seed noise, 5-seed stats, crossed seeds × events CIs |
| `doc_tables.py` | 5-seed tables for `docs/lifecycle_report.md` |
| `run_veracity.sh` | 5-seed veracity verifiers + RQ pipeline |

Artifacts are under `artifacts/generalization/`.

## TL;DR

1. **Multi-seed baseline.** 4-class DGTR pooled macro-F1 is **0.369 ± 0.016**; binary is **0.723 ± 0.008**.
   - A `gat_full` bug was found and fixed: it ignored `--seed`. A same-seed DGTR rerun differs from the committed run by 0.02 macro-F1, so single-seed differences below about 0.02 are noise.
2. **Event-cue masking does not help DGTR**, on either task.
   - 4-class: Δ −0.008 ± 0.020 (2/5 seeds).
   - Binary: Δ −0.004 ± 0.015 (2/5 seeds).
   - It gives small, inconsistent gains to the static models. The event remains highly identifiable after masking: a linear probe recovers it with 0.82 accuracy, against chance ≈ 0.11.
3. **End-to-end BERTweet fine-tuning → Future Work** (infeasible on this machine). The full grid would take about 150 GPU-hours on an M2. The code is ready (`finetune.py`) if a CUDA GPU becomes available.
4. **Event-adversarial training → Future Work** (gate not met). The pre-registered gate is "only if masking helps", and masking did not help. The gradient-reversal head is implemented in DGTR (`--adv-lambda`) and covered by unit tests.
5. **Best configuration = baseline.** Rerunning RQ1–RQ3 over 5 seeds:
   - RQ1's direction holds, but the "no detectable loss" wording does not survive.
   - RQ2 holds.
   - **RQ3's detection advantage does not survive.** d − c F1 is +0.10 ± 0.09, CI [−0.08, +0.28], and the same-seed noise is 0.11.
   - The full table is `docs/claims_survival.md`.
   - **Accuracy does not meaningfully rise across snapshots:** +0.95 pp from 10% to 100%, with macro-F1 flat.

## 1. Multi-seed baseline (reference)

**Bug fixed.** `run_verifiers.py` never passed `--seed` to `gat_full`, so every seed trained it with seed 7. It was rerun per seed (`--merge` updates only that model). `source_lr` and `prefix_lr` are deterministic (lbfgs), so their std is 0 by construction.

**Single-run noise.** The committed seed-7 DGTR (pooled 4-class macro-F1 **0.358**) and a same-seed rerun on this branch (**0.377**) differ. Training is on CPU with fixed seeds, so the likely cause is non-deterministic multithreaded scatter in GATConv, amplified by validation epoch selection. `gat_full` and the LR baselines reproduce exactly.

### Pooled LOEO results (mean ± std over 5 seeds)

| task | model | macro-F1 @10% | macro-F1 @100% | fold-mean macro-F1 @100% | accuracy @100% |
|---|---|---|---|---|---|
| 4-class | **dgtr** | 0.369 ± 0.011 | **0.369 ± 0.016** | 0.231 ± 0.008 | 0.603 ± 0.004 |
| 4-class | gat_full | 0.345 ± 0.006 | 0.341 ± 0.006 | 0.231 ± 0.013 | 0.548 ± 0.010 |
| 4-class | source_lr | 0.321 | 0.321 | 0.213 | 0.540 |
| 4-class | prefix_lr | 0.311 | 0.313 | 0.211 | 0.546 |
| binary | **dgtr** | 0.723 ± 0.010 | **0.723 ± 0.008** | 0.576 ± 0.021 | 0.745 ± 0.008 |
| binary | gat_full | 0.677 ± 0.008 | 0.677 ± 0.006 | 0.552 ± 0.005 | 0.698 ± 0.007 |
| binary | source_lr | 0.696 | 0.696 | **0.580** | 0.716 |
| binary | prefix_lr | 0.667 | 0.678 | 0.561 | 0.704 |

**Pooled vs per-event.** DGTR beats `source_lr` on pooled binary macro-F1 in 5/5 seeds (+0.027). It does so in only **5/9 held-out events** (Wilcoxon p = 0.82), and its fold-mean is lower. The pooled advantage comes from charliehebdo and sydneysiege, the two largest events.

### Per-event DGTR macro-F1 @100% (mean ± std over seeds)

| event | 4-class | binary |
|---|---|---|
| charliehebdo | 0.375 ± 0.021 | 0.760 ± 0.025 |
| ebola-essien | 0.013 ± 0.028 | 0.368 ± 0.037 |
| ferguson | 0.247 ± 0.022 | 0.547 ± 0.075 |
| germanwings-crash | 0.297 ± 0.027 | 0.622 ± 0.056 |
| gurlitt | 0.173 ± 0.030 | 0.483 ± 0.068 |
| ottawashooting | 0.373 ± 0.012 | 0.724 ± 0.028 |
| prince-toronto | 0.046 ± 0.014 | 0.336 ± 0.057 |
| putinmissing | 0.196 ± 0.033 | 0.607 ± 0.034 |
| sydneysiege | 0.363 ± 0.025 | 0.737 ± 0.019 |

The per-seed tables and the other models are in `artifacts/generalization/baseline/summary.json`.

## 2. Event-cue masking

**What changed relative to the existing embeddings.** The original BERTweet normalisation already maps @mentions to `@USER` and URLs to `HTTPURL`. The new masking adds two things:
- hashtags → `HASHTAG` (30,404 replacements);
- spaCy `en_core_web_lg` entities:
  - PERSON → `PERSON` (13,817);
  - GPE / LOC / FAC → `LOCATION` (16,425);
  - ORG → `ORGANIZATION` (14,279).

It covers all 104,582 posts. Embeddings were regenerated with the identical pipeline: same revision, mean pooling and 128-token limit. Reproducing 500 unmasked rows gives cosine **1.00000**, so any difference comes from the masking alone. Masking is per-post and label-free, so it cannot leak across folds.

**The model choice matters.** `en_core_web_sm` missed names such as "Andreas Lubitz" and "Haron Monis"; `lg` catches them. Both miss some event names, e.g. "Germanwings".

### Event-identifiability probe

A 5-fold linear probe predicts the event (9 classes; chance ≈ 0.11):

| embeddings | from source tweet | from thread mean |
|---|---|---|
| original | acc 0.989 / macro-F1 0.986 | 0.966 / 0.947 |
| masked | acc 0.817 / macro-F1 0.843 | 0.865 / 0.865 |

Masking removes names, but topic words ("hostage", "crash", "cartoonists", "shooting") still identify the event.

### Masked vs baseline (paired, 5 seeds, @100%)

| task | model | Δ pooled macro-F1 (mean ± std) | seeds Δ>0 | folds Δ>0 | Wilcoxon p |
|---|---|---|---|---|---|
| 4-class | **dgtr** | **−0.008 ± 0.020** | 2/5 | 4/9 | 0.82 |
| 4-class | gat_full | +0.014 ± 0.009 | 5/5 | 5/9 | 0.65 |
| 4-class | prefix_lr | +0.014 | 5/5 | 6/9 | 0.36 |
| 4-class | source_lr | +0.019 | 5/5 | 6/9 | 0.20 |
| binary | **dgtr** | **−0.004 ± 0.015** | 2/5 | 7/9 | 0.25 |
| binary | gat_full | +0.009 ± 0.008 | 4/5 | 6/9 | 0.16 |
| binary | prefix_lr | −0.011 | 0/5 | 5/9 | 0.55 |
| binary | source_lr | −0.004 | 0/5 | 5/9 | 1.00 |

**Per-class F1 @100% (4-class, mean over seeds).**
- DGTR barely moves: false 0.190 → 0.190, true 0.375 → 0.342, unverified 0.113 → 0.118, non-rumour 0.796 → 0.791.
- The LR gains are almost all on *true*: source_lr 0.230 → 0.323, prefix_lr 0.198 → 0.290. They are paid for by *non-rumour*.

**Verdict: masking does not help** by the pre-registered rule, which requires that for DGTR on either task the mean Δ > 0, ≥ 4/5 seeds improve and ≥ 6/9 events improve. The two DGTR cases fail differently:
- **4-class:** no improvement.
- **Binary:** seed-averaged per-event Δ is positive in 7/9 events (fold-mean +0.018, p = 0.25). The two largest events, sydneysiege (−0.055) and germanwings (−0.032), dominate the pooled score. This is suggestive at best.

## 3. Fine-tuning and event-adversarial training → Future Work

Neither was run. Both are implemented; the reasons are given under **Future work** at the end of this document.

## 4. RQ1–RQ3 for the best configuration (= baseline) over 5 seeds

The definitive claim-by-claim verdicts live in **`docs/claims_survival.md`**. That file gives the committed value, the 5-seed mean ± std, the CI from a crossed bootstrap over seeds × held-out events, the same-seed noise and a verdict for every claim. This section summarizes it.

| claim | committed | 5 seeds | CI (seeds × events) | same-seed noise | verdict |
|---|---|---|---|---|---|
| RQ1: always-10% − always-100% accuracy | −0.67 pp | −0.95 ± 0.17 pp | [−1.55, +0.06] pp | 0.08 pp | weaker (≈ 1 pp; not established across events) |
| RQ1: calibration-aware threshold − final-only | −0.03 pp | −0.53 ± 0.13 pp | [−0.93, −0.08] pp | 0.28 pp | **"no detectable loss" doesn't survive** |
| RQ2: final accuracy d − b | −0.12 pp | −0.16 ± 0.14 pp | [−0.49, +0.02] pp | 0.03 pp | survives (no change established) |
| RQ2: utility w0.5 d − b | −0.0039 | −0.0047 ± 0.0014 | [−0.0123, −0.0003] | 0.0013 | survives |
| RQ3: reopen F1 d − c (4-class) | +0.135 | +0.10 ± 0.09 | [−0.08, +0.28] | 0.108 | **doesn't survive** |
| RQ3: reopen F1 d − c (veracity) | +0.138 | +0.10 ± 0.11 | [−0.06, +0.25] | 0.0001 | **doesn't survive** |
### Per seed

| seed | always10 − always100 | calibrated − final | RQ2 acc d − b | RQ3 F1 d − c |
|---|---|---|---|---|
| 7 | −0.75 [−1.31, −0.22] | −0.31 [−0.70, +0.09] | −0.09 [−0.47, +0.25] | +0.243 [+0.215, +0.271] |
| 13 | −0.81 [−1.34, −0.26] | −0.56 [−0.90, −0.22] | −0.11 [−0.26, +0.05] | +0.108 [+0.077, +0.139] |
| 21 | −1.06 [−1.57, −0.51] | −0.62 [−1.00, −0.25] | −0.40 [−0.73, −0.08] | +0.084 [+0.052, +0.117] |
| 42 | −1.15 [−1.67, −0.64] | −0.64 [−0.97, −0.33] | −0.14 [−0.26, −0.02] | −0.020 [−0.051, +0.011] |
| 99 | −0.97 [−1.43, −0.51] | −0.51 [−0.84, −0.17] | −0.03 [−0.23, +0.14] | +0.093 [+0.060, +0.124] |

### Does accuracy now rise across snapshots?

**No, not meaningfully, in any configuration.** Pooled held-out values, mean over 5 seeds:

| task | config | model | accuracy 10% → 100% | Δ accuracy | Δ macro-F1 |
|---|---|---|---|---|---|
| 4-class | baseline | dgtr | 0.593 → 0.603 | +0.95 ± 0.17 pp (5/5) | −0.001 ± 0.006 |
| 4-class | masked | dgtr | 0.575 → 0.592 | +1.75 ± 0.35 pp (5/5) | +0.001 ± 0.003 |
| 4-class | masked | gat_full | 0.547 → 0.558 | +1.07 ± 0.34 pp (5/5) | +0.001 ± 0.004 |
| binary | baseline | dgtr | 0.742 → 0.745 | +0.28 ± 0.20 pp | −0.000 ± 0.002 |
| binary | masked | dgtr | 0.743 → 0.743 | +0.03 ± 0.47 pp | −0.005 ± 0.006 |

**Masking roughly doubles the 4-class rise in accuracy**, but only because it **lowers the starting point**. With source-tweet event cues removed, early accuracy drops 1.8 pp and later replies recover part of it. Final accuracy is still lower than the baseline's.

Macro-F1 is flat across snapshots everywhere. The RQ1 framing, "the signal is in the source tweet and the first replies", is unchanged.

## 5. Error analysis

Sources: `artifacts/generalization/rq/baseline/seed7/4class/error_analysis.md` and `per_event.json`; the per-event 5-seed table is in §1.

### Where it fails

**Label-prior shift between events.** prince-toronto (98% rumours, almost all *false*) and ebola-essien (14 threads, all *false*) get 4-class macro-F1 ≈ 0.01–0.05. The training events are 55–65% non-rumour, and the verifier predicts the training prior. Accuracy on these events is below majority-class, which is the per-fold failure documented in `docs/rq_results.md`. None of the representation changes tried here move these events, because the problem is a label-prior shift, not an event-vocabulary cue.

**Validation-selected commit thresholds can disable the lifecycle on an event.** In the charliehebdo fold, validation on germanwings selected τ_prov = 0.8 / τ_conf = 0.95. On charliehebdo's 2,079 test threads, system (d) then **never commits** (final coverage 0.00).

**Biggest events dominate pooled scores.** charliehebdo (2,079 threads) and sydneysiege (1,221) together are 51% of all threads. Pooled and fold-mean macro-F1 can therefore rank models differently (§1, §2).

### Three trajectories (system d, baseline seed 7, picked by a fixed rule in `error_analysis.py`)

**1. Correct early commit.** sydneysiege, gold *true*, 9 posts.
- The source is "LIVE: Sydney cafe siege, hostages held in CBD, reporter … at scene".
- The verifier gives p(true) = 0.70 on the source alone.
- It commits *true* at 20% (2 min, 2 posts) and confirms at 30% (3 min).
- Conflict stays ≈ 0 throughout: the replies are sympathy and sharing, with no stance.
- The decision was made from the source tweet; replies added nothing, which is the RQ1 pattern.

**2. Wrong early commit.** prince-toronto, gold *false*, 14 posts.
- The source is the Live Nation apology.
- The verifier says *non-rumour* with p = 0.88 from the first post, and that never moves.
- It commits at 20% and confirms at 30%.
- The replies push back ("Why did you wait until 2:45pm … to announce there was no show?"):
  - conflict strength reaches **0.99**;
  - net conflict climbs to **1.21** over 5 distinct clusters by 100%.
- **No reopen**, because validation selected the *legacy* gate with θ_net = 2.0. With CONFIRMED hysteresis (κ = 1.5), a reopen needs net ≥ 3.0.
- The conflict detector saw the contradiction, and the validation-selected thresholds made the gate inert. This is the RQ3 recall problem in one example.

**3. A reopen (justified and corrected).** sydneysiege, gold *true*, 29 posts.
- The source is the photo of a fleeing Lindt employee.
- At 20% (6 posts) it commits *false* with p = 0.43.
- At 30% (9 posts, 4 min) three distinct contradicting clusters arrive (strength 0.94, volume 3), and the gate reopens.
- Re-verification immediately moves to *true* (p = 0.50).
- It stays PROVISIONAL *true* to the end and never reaches CONFIRMED (p < τ_conf), which is correct.

## 6. Why cross-event generalization is hard

This is the summary; the full version is `docs/lifecycle_report.md` §6.

1. **The event is recoverable from the text.** A linear probe identifies the event at 0.99 from source embeddings, and still at **0.82** after masking hashtags and named entities (chance ≈ 0.11). Topic words carry it.
2. **Masking does not help DGTR** (§2). The static models' gains are not consistent across held-out events.
3. **The SAMGAT protocol comparison** (branch `samgat-diagnosis`, commit 6acff67) runs the same GAT+BERTweet on the binary task:
   - random stratified 5-fold, with events mixed across train and test: macro-F1 **0.831** (0.848 with test-fold epoch selection; the paper reports 0.865);
   - LOEO: **0.673** (5-seed rerun: 0.677 ± 0.006).

   About 0.16 macro-F1 depends on having seen the test event. That is the event-specific signal the probe detects.
4. **Label priors shift by event:** the non-rumour share runs from 0.00 to 0.78, while training events are 55–65%. Representation changes cannot fix a prior shift.
5. **Two events are 51% of all threads,** so pooled and per-event conclusions can disagree.

## What changed in the code (all backwards-compatible; the baseline path is unchanged)

- **`experiments/run_verifiers.py`:**
  - `gat_full` now receives `--seed` (bug fix);
  - new flags: `--embedding-dir`, `--threads`, `--merge` (update only the given models in existing fold files), `--skip-done` and `--adv-lambda`;
  - binary-task `num_classes` (pre-existing uncommitted fix).
- **`lifecycle/verifier.py`:** optional gradient-reversal event head; default off, so logits are identical when off (tested).
- **`experiments/run_injection.py`:** `--verifier-embedding-dir`, so a DGTR trained on masked embeddings sees masked injected posts. Stance and evidence keep the original embeddings.
- **`experiments/rq_analysis.py`:** `--root` / `--tasks` / `--output`; non-clean variants are optional.
- **`lifecycle/data.py`, `lifecycle/baselines.py`:** binary task (pre-existing uncommitted change).
- **Tests:** 2 new tests in `tests/test_verifier.py`; the full suite passes.


## Future work (not run, and why)

### End-to-end BERTweet fine-tuning with class-balanced loss (with and without masking)

**Reason not run: infeasible on the available hardware.** The full grid needs ≈ 150 GPU-hours on an Apple M2; details below.

**Implemented** (`experiments/generalization/finetune.py`; smoke-tested on one fold):
- **Model:** BERTweet as a causal *prefix* classifier. The input is (source tweet, replies observed so far, oldest first); nothing after the prefix is read.
- **Loss:** class-balanced cross-entropy (Cui et al. 2019, β = 0.999).
- **Training and selection:** one random checkpoint prefix per thread per epoch; the epoch is selected on validation-event macro-F1.
- **Options:** original or masked text, and an optional gradient-reversal event head.
- **Output:** the same fold-file format as `run_verifiers.py`, so `compare.py` and the lifecycle read it unchanged.

Fine-tuning *inside* the graph model would mean re-embedding all 104k posts for every fold, which was ruled out first.

**Measured on Apple M2 (MPS, bf16):**
- 4.6 s per training step (batch 32, 128 tokens), or 2.3 s at 64 tokens;
- 2.4 s per 64-sequence inference batch.

That gives about 12 min per epoch on about 5k training threads and **≈ 50 min per fold**. The requested grid is 5 seeds × 9 folds × 2 tasks × 2 text variants = 180 folds ≈ **150 h**. Even 1 seed × 1 task × 64 tokens × 2 epochs is about 5 h per text variant.

**Feasible later.** On a single A100 this is roughly 10–15× faster, about 10–15 h for the full grid. The command is in the module docstring.

### Event-adversarial training (gradient reversal on an event classifier)

**Reason not run: the pre-registered gate "only if masking helps" was not met** (§2; `artifacts/generalization/DECISIONS.md`). The case for overriding the gate is that the event is still 82% identifiable after masking (§6).

The spec says "only if masking helps", and §2 shows it did not.

**Implemented anyway**, so it costs nothing to try later:
- DGTR has an optional gradient-reversal event head (`VerifierConfig.num_events` / `adv_lambda`; `run_verifiers.py --adv-lambda`). It uses the DANN λ schedule on the final temporal representation, over *training* events only.
- Inference is unchanged.
- Unit tests in `tests/test_verifier.py` check three things: the gradient sign flips; logits match the plain model; and validation/test threads never reach the event head.

The probe in §2 is the strongest argument for still trying it: masking leaves the event 82% identifiable. This would need a deliberate decision to override the gate.
