# When to commit, and when to take it back: decision timing for early rumour verification

**One-page summary · PHEME 4-class, 9-event leave-one-event-out · clean protocol: every selection is made on validation events, test events are scored once · 5 seeds; mean ± std, with 95% CIs over seeds × held-out events**

## Question
Rumour classifiers are usually judged on one prediction made after a conversation ends. A deployed system has to decide *during* the conversation: wait for more replies, or commit now.

Our core question is **decision timing**. Revision is treated as a **safety layer**: it should leave ordinary decisions alone and withdraw a commitment when meaningful contradicting evidence arrives late.

## Setup
- **Snapshots:** causal snapshots at 10/20/30/50/75/100% of each thread's replies; a leakage test confirms future posts never affect earlier predictions.
- **Verifier:** a dynamic-graph verifier (graph attention over BERTweet reply vectors, then a causal temporal transformer), temperature-calibrated per fold.
- **Commit policies** compared on validation events with cross-fitting:
  - fixed checkpoint;
  - raw-confidence threshold;
  - calibration-aware threshold;
  - cost-sensitive stopping rule.
- **Safety layer:**
  - a reply-stance model (RumourEval labels; features chosen on validation events);
  - a conflict detector separating one strong contradiction from many weak ones and collapsing duplicates;
  - a WAIT → PROVISIONAL → CONFIRMED → REOPEN state machine whose gate design and thresholds are chosen on validation events.
- **Robustness:** all results are rerun over 5 seeds.
  - Training the verifier twice with the *same* seed moves macro-F1 by 0.019 and accuracy by 2.4 pp.
  - Differences smaller than that are treated as not established.
  - Each claim's verdict is in `docs/claims_survival.md`, with claim IDs in brackets below.

## Findings

**RQ1. Later propagation adds little on PHEME.**
- The verifier's held-out accuracy barely moves as a thread grows: 0.593 at 10% → 0.603 at 100%, and macro-F1 stays flat at 0.369 [V6].
- Committing every decision at the 10% checkpoint costs **≈ 1 pp** relative to waiting for the full thread. That is −0.95 ± 0.17 pp, negative in every seed; across held-out events the CI reaches 0, [−1.55, +0.06] pp [T1].
- An untuned "always commit at 10%" policy **ties all four tuned commit policies** on earliness-weighted utilities [T2].
- A calibration-aware threshold commits on 90% ± 3% of threads within the first fifth of the conversation, a median of about 3 minutes after the source [T4]. It **costs ≈ 0.5 pp**, −0.53 ± 0.13 pp, CI [−0.93, −0.08], relative to waiting [T5].
- The signal is in the source tweet and the first few replies.

**RQ2. Revision does not measurably harm ordinary decisions.**
- Adding conflict-gated revision changes final accuracy by **−0.16 ± 0.14 pp**, CI [−0.49, +0.02], which is not established [H1].
- Harmed threads slightly outnumber helped ones in every seed [H2].
- It costs a small amount of timing utility, −0.005 [−0.012, −0.0003], because a reopen withdraws a decision [H3].

**RQ3. Detecting injected late contradictions: not established.**
- The test injects real stance-labelled replies at the end of threads that already hold a decision.
- Conflict-gated revision reaches **reopen F1 0.20 ± 0.08** vs 0.10 ± 0.02 for revising on prediction changes.
- The paired Δ is **+0.10 ± 0.09**: it ranges from −0.02 to +0.24 across seeds, with CI [−0.08, +0.28].
- The same-seed noise on this Δ is 0.11, as large as the effect itself [D1].
- Recall is low: 0.13 ± 0.06 [D3].
- What detection there is comes from the folds where validation picked the less conservative gate. Which folds those are (2–5 of 9) depends on the seed [D4].

## Limits (stated up front)
1. **Below the majority-class baseline on accuracy in most events.**
   - The verifier is less accurate than always predicting "non-rumour" in 6–7 of 9 held-out events in every seed.
   - Pooled, it scores 0.603 vs 0.626; that gap is within noise [V5].
   - It clearly beats the majority baseline only on macro-F1: 0.369 vs 0.193 [V4].
   - All timing claims are therefore *relative*: early vs late decisions of the same model.
2. **Veracity (true / false / unverified) is out of scope as a result.**
   - It is below the majority baseline in 4–5 of 9 events, and pooled (0.329 vs 0.444) in every seed [V10].
   - Its accuracy is at chance level for 3 classes [V9].
3. **PHEME carries little late-contradiction signal.**
   - Even human-annotated reply stance shifts veracity odds by at most about 1.7× per reply [S2].
   - Oracle stance does not raise utility [H8].
4. **The injection test shows detection, not correctness.** It tests whether the layer *detects and withdraws* contradicted decisions, not whether the revised label is right. Even detection is not established under seed noise [S3, D1].
5. **The new temporal verifier ties the project's existing GAT model** on both 4-class and veracity; it does not beat it [V2, V8].
6. **Single runs are not reliable on this pipeline.**
   - The earlier one-seed numbers suggested two things that do not survive 5 seeds:
     - a calibration-aware commit rule with "no measurable cost" (T5);
     - a clear detection advantage of +0.14 (D1).
   - Two earlier selection leaks (gate design and stance features chosen using test-event information) remain fixed.

## Why cross-event generalization is hard
- **The event is recoverable from the text.** A linear probe identifies the event from a source tweet with 99% accuracy (chance ≈ 11%). After masking hashtags, names, places and organisations it still reaches **82%**: topic words give the event away.
- **Masking those cues does not help the verifier.** Macro-F1 changes by −0.008 on 4-class and −0.004 on binary, not established.
- **The same GAT+BERTweet model is much better when the test event is seen in training.** On the binary task it scores macro-F1 0.83 on SAMGAT's random split, where events are mixed, vs 0.67 under leave-one-event-out. The published gap is protocol, not architecture.
- **Label mixes differ sharply between events.** The non-rumour share runs from 0% to 78%, and a better text representation cannot correct for that.

## Future work
- **End-to-end BERTweet fine-tuning with class-balanced loss.** It is implemented but not run. It needs about 150 GPU-hours on the available Apple M2 (about 50 min per fold), or roughly 10–15 h on one A100.
- **Event-adversarial training (gradient reversal).** It is implemented and tested but not run. The pre-registered condition, "only if masking helps", was not met. The 82% probe result is the argument for trying it anyway.
- **Deterministic training, or ≥ 5 seeds by default,** before any RQ3-style claim is made.

## Contribution in one sentence
A reproducible, leak-checked, multi-seed protocol showing that on PHEME almost all usable signal arrives in the first tenth of a conversation, so a calibrated rule can commit within the first fifth at a cost of about half a point of accuracy. A validation-selected revision layer can be added without measurably changing those decisions, but its advantage at detecting injected late contradictions is not established under seed noise. Absolute accuracy remains below a majority-class baseline in most events, largely because events differ in topic and label mix in ways a model trained on other events cannot see.
