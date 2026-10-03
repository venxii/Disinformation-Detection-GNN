# When to commit, and when to take it back: decision timing for early rumour verification

**One-page summary · PHEME 4-class, 9-event leave-one-event-out · clean protocol: every selection is made on validation events, test events are scored once**

## Question
Rumour classifiers are usually judged on one prediction made after a conversation ends. A deployed system has to decide *during* the conversation: wait for more replies, or commit now. Our core question is **decision timing**. Revision is treated as a **safety layer**: it should leave ordinary decisions alone and withdraw a commitment when meaningful contradicting evidence arrives late.

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

## Findings

**RQ1. Later propagation adds little on PHEME.**
- The verifier's accuracy barely moves as a thread grows (held-out 0.568 at 10% → 0.575 at 100%).
- Committing every decision at the 10% checkpoint costs **0.67 pp [0.14, 1.17]** relative to waiting for the full thread.
- An untuned "always commit at 10%" policy **ties all four tuned commit policies** on earliness-weighted utilities.
- A calibration-aware threshold commits on 88% of threads within the first fifth of the conversation (median about 3 minutes after the source), with **no detectable loss** relative to waiting: −0.03 pp [−0.42, +0.34].
- The signal is in the source tweet and the first few replies.

**RQ2. Revision does not harm ordinary decisions.**
- Adding conflict-gated revision changes final accuracy by **−0.12 pp [−0.47, +0.20]**. Harmed and helped threads roughly cancel (65 vs 57 of 6,424).
- It costs a small amount of timing utility (−0.004 [−0.007, −0.0005]), because a reopen withdraws a decision.

**RQ3. Revision catches injected late contradictions better than reacting to prediction changes, but only partly.**
- Real stance-labelled replies were injected at the end of threads that already hold a decision.
- **Reopen F1: 0.24 vs 0.11**, paired Δ **+0.14 [+0.10, +0.17]**.
- Recall is low (0.15).
- In 7 of 9 folds, validation selected a conservative gate that almost never reopens. The advantage comes from the other 2 folds.

## Limits (stated up front)
1. **Below the majority-class baseline on accuracy.** The verifier is less accurate than always predicting "non-rumour" in 6 of 9 held-out events (pooled 0.575 vs 0.626). It beats that baseline only on macro-F1 (0.358 vs 0.193). All timing claims are *relative*: early vs late decisions of the same model.
2. **Veracity (true / false / unverified) is out of scope as a result.** It is below the majority baseline in 4 of 9 events and pooled (0.285 vs 0.444).
3. **PHEME carries little late-contradiction signal.** Even human-annotated reply stance shifts veracity odds by at most about 1.7× per reply, so revision cannot raise natural accuracy.
4. **The injection test shows detection, not correctness.** It shows that the layer *detects and withdraws* contradicted decisions. It does not show that the revised label is right.
5. **The new temporal verifier ties the project's existing GAT model**; it does not beat it.
6. **Two earlier selection leaks were fixed.** The gate design and stance features had been chosen using test-event information. Under the clean protocol the timing and no-harm results are unchanged, and the detection advantage shrinks from +0.21 to +0.14.

## Contribution in one sentence
A reproducible, leak-checked protocol showing that on PHEME almost all usable signal arrives in the first tenth of a conversation, so a calibrated rule can commit within the first fifth at no measurable cost relative to waiting. A validation-selected revision layer can be added without harming those decisions and detects injected late contradictions better than prediction-change revision. Absolute accuracy remains below a majority-class baseline, and natural PHEME threads contain too little contradicting signal for revision to raise it.
