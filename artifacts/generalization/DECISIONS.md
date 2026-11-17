# Pre-registered decision rules (written 2026-11-17 19:35 IST, before the full step-2 results)

At writing time only 3 of 5 seeds of 4-class masked results existed and had been looked at
(DGTR Δ −0.005, gat_full +0.009, LR +0.014/+0.019 pooled macro-F1). No binary masked results had been seen.

**"Masking helps" (gates step 4).** For DGTR (the verifier that RQ1–RQ3 run on), on at least one task
(4class or binary), at the 100% checkpoint:
- mean paired Δ pooled LOEO macro-F1 > 0, **and**
- Δ > 0 in ≥ 4 of 5 seeds, **and**
- seed-averaged Δ > 0 in ≥ 6 of 9 held-out events.

If DGTR fails but gat_full passes, masking is reported as helping only the static GAT, and step 4 is not run
(the lifecycle runs on DGTR).

**"Best configuration" for step 5.** The configuration with the highest mean pooled 4-class DGTR macro-F1 @100%
among those that pass the rule above; otherwise the baseline (and step 5 then reports the 5-seed baseline RQs,
which still answers whether the conclusions hold across seeds).

**Step 3 (fine-tuning).** Stopped as infeasible: measured 4.6 s per training step (batch 32, 128 tokens, bf16 on
Apple M2 MPS) → ≈ 50 min per fold → ≈ 150 h for 5 seeds × 9 folds × 2 tasks × 2 masking variants.
