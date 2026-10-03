# Evidence-Aware Early Rumor Verification: Reconnaissance Audit

**Status:** first-phase audit, before substantial model implementation  
**Dataset in this checkout:** PHEME-derived processed data  
**Scope:** established methods, reusable implementations, architecture, gap, plan, and risks

## Executive conclusion

PHEME already supports the core research question, but the existing checkout is currently a **static graph export** rather than a temporally causal evaluation system. The defensible capstone is therefore:

1. retain the existing PHEME extraction and validation pipeline after correcting/confirming its provenance;
2. add a reproducible event-held-out split and snapshot builder that only exposes posts available at each observation point;
3. compare text-only, graph-only, text+graph, and text+graph+snapshot-temporal baselines;
4. calibrate a small selective controller that emits `WAIT`, `PROVISIONAL`, or `CONFIRMED`; and
5. represent later contradictory evidence as a measurable **reopen event**, followed by re-evaluation on the updated snapshot.

The central research contribution should be evaluated as **decision quality under partial and changing evidence**, not as a new GNN layer.

## A. Existing-method map

| Component | Existing method | Paper / source | Existing implementation | Decision | Reason |
| --- | --- | --- | --- | --- | --- |
| PHEME data model | Source tweet plus reaction conversation, rumor/non-rumor and true/false/unverified annotations | [PHEME project resources](https://www.pheme.eu/software-downloads/); Zubiaga et al., PHEME dataset | Existing `dataset/preprocess.py`; official PHEME resources | **REUSE, VERIFY** | The checkout already converts source/reaction JSON, annotations, and nested structure into tabular data. Raw-data availability and exact version must be recorded. |
| Propagation graph | Conversation propagation tree / graph with posts as nodes and replies as edges | [PGNN/GLO-PGNN/ENS-PGNN study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7274137/) | Existing `dataset/propagation.py`, `dataset/graph_export.py`; PyTorch Geometric `Data` is the target adapter | **REUSE, ADAPT** | Existing edges are useful, but graphs must be induced by an observation cutoff; full-thread trees must not be used for early snapshots. |
| Text representation | Pretrained social-media language encoder, especially BERTweet | Nguyen et al., [BERTweet](https://aclanthology.org/2020.emnlp-main.721/) | [Hugging Face BERTweet model family](https://huggingface.co/vinai/bertweet-base) and `transformers` | **REUSE** | Tokenization and transformer modeling are mature. Freeze embeddings first to keep the capstone computationally tractable; fine-tuning is an optional later ablation. |
| Graph baseline | GCN/GAT or bidirectional graph convolution over conversation trees | [Bi-GCN](https://aclanthology.org/2020.acl-main.245/); [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) | PyTorch Geometric `GCNConv`, `GATConv`, pooling, loaders | **REUSE** | There is no research justification for implementing graph layers from scratch. A small GAT/GCN baseline is easier to reproduce and defend. |
| Early rumor detection | Partial cascades evaluated by tweet count and/or time deadline | [PGNN early stopping results](https://pmc.ncbi.nlm.nih.gov/articles/PMC7274137/); [TGAN](https://research.cuhk.edu.hk/en/publications/tgan-temporal-aware-graph-attention-network-for-early-rumor-detec/) | Existing research code varies in maturity; PyG batching plus our snapshot builder | **ADAPT** | The requested 10/20/30/50/75/100% node snapshots and elapsed-time windows need a common, causal evaluator. |
| Temporal graph information | Time-aware graph attention and temporal propagation features | [TGAN](https://research.cuhk.edu.hk/en/publications/tgan-temporal-aware-graph-attention-network-for-early-rumor-detec/); [temporal propagation optimization](https://aclanthology.org/2025.coling-main.261.pdf) | Existing `dataset/temporal_features.py`; PyG edge attributes / ordinary feature tensors | **ADAPT** | Recompute duration, delay, depth, and counts on each prefix. Full-conversation temporal aggregates are future leakage at early checkpoints. |
| Propagation uncertainty | Bayesian / edge-enhanced Bayesian GCN for rumor detection | [EBGCN](https://aclanthology.org/2021.acl-long.297/) | Research implementations exist but are not yet verified in this checkout | **ADAPT, OPTIONAL** | A Bayesian GNN is a credible uncertainty baseline, but it adds substantial complexity. Use calibrated predictive probabilities as the primary controller input and add EBGCN only if time permits. |
| Probability calibration | Post-hoc calibration such as temperature scaling; reliability diagrams and proper scoring rules | Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | `sklearn` metrics/calibration utilities; temperature scaling can be a small adapter | **REUSE, BUILD ADAPTER** | The adapter is glue around an established method, not a new calibration algorithm. Calibrate on validation events only. |
| Selective prediction / abstention | Risk-coverage evaluation and selective classification | Geifman & El-Yaniv, [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) | [MAPIE](https://mapie.readthedocs.io/) for conformal prediction and coverage-oriented workflows; scikit-learn metrics | **REUSE, ADAPT** | `WAIT` is an abstention action. Thresholds must be selected on validation data and reported with coverage, selective risk, and unnecessary-wait rates. |
| Distribution-free uncertainty | Split conformal prediction / prediction sets | [MAPIE documentation](https://mapie.readthedocs.io/); Angelopoulos & Bates, [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511) | MAPIE | **OPTIONAL REUSE** | Conformal sets offer interpretable uncertainty, but exchangeability is strained by sequential snapshots. Use only with an explicit limitation statement, or keep calibrated abstention as the main controller. |
| Prediction stability | Sequential consistency / change tracking across checkpoints | No single established rumor-specific standard identified; sequential prediction stability is a measurement protocol | **Build small evaluator** | **BUILD** | The project needs a trajectory table: label, probability, confidence, checkpoint, change, and stability. This is experiment orchestration rather than a novel model. |
| Contradictory evidence | Evidence-aware representation and uncertainty are studied, but a standard PHEME protocol for late contradictory replies and reopening was not found in the reviewed sources | EBGCN addresses propagation uncertainty; early-detection papers generally stop at a deadline or final snapshot | No verified drop-in implementation found | **BUILD, narrowly** | Implement contradiction as a measurable later-snapshot event using a held-out signal (e.g., text stance/semantic opposition or a validated annotation rule), then apply a thresholded reopen policy. Do not claim a general fact-checking system. |
| Online revision | Dynamic/evolving rumor detection papers update predictions as cascades grow | [Early detection based on propagation prediction](https://link.springer.com/article/10.1007/s40747-025-02140-z) | No mature, dataset-compatible revision controller verified | **ADAPT** | The model can simply be called again on every new snapshot. The state machine around those calls is the capstone-specific adapter. |

### What was not found

The reviewed literature contains substantial early rumor detection, temporal graph modeling, and uncertainty modeling. It does **not** provide a clearly standardized, reusable PHEME benchmark in which a high-confidence provisional decision is explicitly reopened after a materially later contradictory observation. That is a promising gap, but it must be stated narrowly: the project can contribute a reproducible evaluation protocol and controller, not a claim to solve factual contradiction in the open world.

## B. Minimal architecture proposal

```text
PHEME raw JSON
  -> existing extraction + validation
  -> event-held-out train/validation/test split
  -> causal snapshot builder
       - node-count prefixes: 10/20/30/50/75/100%
       - elapsed-time prefixes: fixed quantiles or fixed deadlines
       - induced reply edges only among available nodes
       - snapshot-local temporal/structural features
  -> four comparable predictors
       1. text-only: pooled BERTweet source/thread representation
       2. graph-only: structural features + GAT/GCN
       3. text + graph: BERTweet node embeddings + GAT/GCN
       4. text + graph + temporal: add snapshot-local delays/depth/counts
  -> validation-fitted calibration
  -> evidence controller
       - calibrated confidence / margin
       - evidence quantity
       - stability across prior snapshots
       - contradiction signal
       - minimum-evidence and hysteresis rules
       - WAIT / PROVISIONAL / CONFIRMED
  -> continue scoring later snapshots
  -> contradiction gate
       - detect material disagreement with provisional state
       - REOPEN
       - score updated snapshot
       - revised PROVISIONAL or CONFIRMED decision
```

### Controller semantics

Use an explicit finite-state record per thread:

`UNDECIDED -> WAIT -> PROVISIONAL(label) -> CONFIRMED(label)`

and, when a later snapshot passes the contradiction/revision gate:

`PROVISIONAL(label) -> REOPENED -> WAIT or PROVISIONAL(new_label)`

The initial implementation should use transparent validation-fitted rules rather than an opaque learned controller. For example, require minimum evidence, calibrated confidence above a threshold, agreement over two successive checkpoints, and no material contradiction. The exact thresholds must be tuned on validation events and frozen before test evaluation. A provisional decision should not be called confirmed merely because confidence is high at one checkpoint.

## C. Research gap and claims boundary

### Already established

- Rumor verification from propagation structure is established; graph neural models such as Bi-GCN and PGNN-family approaches already use conversation trees.
- Early detection is established using partial cascades, elapsed-time deadlines, tweet counts, and dynamic temporal modeling.
- Text-plus-propagation fusion is established; BERTweet is an appropriate reusable social-media encoder.
- Propagation uncertainty and Bayesian graph models are established, including EBGCN.
- Calibration and abstention/selective prediction are established machine-learning methods.

### Defensible gap

The gap is the **evaluation and decision protocol for evidence-aware revision**: under causally truncated, progressively arriving PHEME evidence, can a calibrated controller abstain early, issue a provisional result only when stable, and explicitly reopen that result when later evidence materially changes the prediction? The study should compare this controller against confidence-only and always-decide baselines.

### Claims to avoid

- Do not call the system a general fact-checker; PHEME labels are conversation-level annotations, not a live truth oracle.
- Do not call a node-count prefix a real-time observation unless its timestamps are also considered.
- Do not call a full-graph score an early prediction.
- Do not infer contradiction solely from a change in model probability; that is prediction instability, not evidence contradiction.

## D. Implementation plan

1. **Dataset and reproducible preprocessing — REUSE/VERIFY.** Record the raw PHEME source/version, run the existing validators, exclude `unknown` from supervised labels unless there is a documented reason to retain it, and add an event-level split manifest. PHEME has no universally accepted official split; leave-one-event-out or event-held-out evaluation is preferable to random thread splitting because event leakage is a major risk.
2. **Baseline reproduction — ADAPT.** Reproduce one published propagation baseline, preferably Bi-GCN or PGNN-family, using the local graph representation and the paper’s compatible label task. Record deviations instead of silently presenting a new implementation as reproduction.
3. **Text-only baseline — REUSE/ADAPT.** Start with frozen BERTweet embeddings and a simple classifier. Use source-only text first; thread-text pooling can be a secondary variant because later replies are unavailable in early snapshots.
4. **Graph baseline — REUSE/ADAPT.** Use PyG GCN/GAT with structural node features and graph pooling. Include a no-edge/single-node path because 684 local threads have no propagation edges according to the project documentation.
5. **Text + graph — ADAPT.** Feed frozen node text embeddings into the graph model and compare against the two baselines under identical splits and snapshots.
6. **Temporal extension — ADAPT.** Add only features available in the current snapshot: relative timestamp, reply delay when parent is available, depth in the induced tree, node count, elapsed duration, and local branching/count features. Avoid the current full-thread aggregates during early evaluation.
7. **Progressive evidence evaluation — BUILD evaluator.** Generate node-count and time-window snapshots, persist a manifest per thread/checkpoint, and output one row per prediction trajectory. Define timestamp ties deterministically and never use the final graph to choose an early cutoff.
8. **Evidence/reliability controller — BUILD small state machine.** Fit calibration on validation events, tune thresholds on validation only, and compare confidence-only, confidence-plus-stability, and full controller variants.
9. **Late contradiction/reopening — BUILD narrow protocol.** Define contradiction before test scoring, log the first qualifying later checkpoint, transition to `REOPENED`, and score the new snapshot. Report how many test threads actually contain such cases; if too few exist, present the result as a stress test or synthetic replay rather than a prevalence claim.
10. **Ablations — REUSE evaluator.** Remove text, structure, temporal features, calibration, stability, and contradiction gating one at a time. Keep the same event split and seeds.
11. **Final evaluation — BUILD reporting.** Report macro-F1 and per-class metrics, calibration error/Brier or log loss, risk-coverage/selective risk, abstention coverage, unnecessary waits, time/tweet count to reliable decision, premature high-confidence error rate, number of revisions, reopen precision/recall, and final accuracy after revisions.

## E. Risk assessment and simplifications

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| PHEME version/schema drift | Repositories often report different counts and tweet recovery states | Freeze a manifest with raw path, checksums if possible, counts, labels, and preprocessing commit. Treat the local CSVs as a derived artifact, not the authority. |
| Event leakage | Randomly splitting threads can let event-specific language and propagation patterns cross splits | Use event-held-out splits, with train/validation/test events selected before model fitting. |
| Future leakage from aggregates | Current `temporal_features.csv` includes last timestamp, total posts, and final depth/duration | Recompute all features inside each snapshot. Retain full-thread features only for the 100% experiment. |
| Label imbalance | Non-rumor dominates; rumor subclasses are smaller | Use macro metrics, class-weighted training only when justified, and report per-class support. Do not collapse labels without an explicit research reason. |
| Deleted/missing tweets | PHEME is an historical Twitter dataset with incomplete recoverability | Preserve missingness indicators and report coverage. Do not silently treat absent posts as negative evidence. |
| Too few late contradictions | PHEME labels do not necessarily mark a timestamped “contradictory reply” | Pre-register an operational definition; count eligible cases. If scarce, use a controlled replay/stress-test and label it accordingly. |
| Calibration under sequential dependence | Snapshot rows from one thread are not independent calibration examples | Split by thread/event first; calibrate on held-out threads, and discuss sequential dependence as a limitation. |
| BERTweet compute | Encoding ~100k posts and repeated snapshots can be expensive | Cache frozen node embeddings once, then reuse them across all snapshot models. |
| Scope creep | Bayesian GNN, conformal prediction, stance, drift, and online learning could overwhelm a capstone | Make calibrated selective prediction the main controller. Treat EBGCN/conformal/stance as optional extensions only after the core experiment works. |
| “Contradiction” ambiguity | A later reply can be skeptical without proving the claim false | Separate a text/stance disagreement signal from the ground-truth label and report both; do not equate disagreement with truth. |

## Local repository findings

The current checkout contains:

- `data/processed/posts.csv`, `claims.csv`, `edges.csv`, `temporal_features.csv`, and `propagation_trees.json`;
- graph JSON exports under `data/graphs/`;
- extraction, tree-building, temporal-feature, export, and validation scripts under `dataset/`;
- a useful existing pipeline description in `docs/data_pipeline.md`.

The documentation reports 6,425 threads, 104,582 posts, 95,074 edges, 4,022 non-rumors, and rumor labels of true 1,067, unverified 697, false 638, plus one unknown. These figures should be regenerated by scripts and included in a versioned audit manifest before model training. The current files expose timestamps and reply IDs, which is sufficient for causal snapshot construction, but the current temporal CSV is full-conversation aggregated and therefore cannot be fed directly to early checkpoints.

## Sources reviewed

- [PHEME project resources](https://www.pheme.eu/software-downloads/)
- [PGNN / GLO-PGNN / ENS-PGNN rumor detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC7274137/)
- [Bi-GCN: rumor detection on social media](https://aclanthology.org/2020.acl-main.245/)
- [EBGCN: propagation uncertainty](https://aclanthology.org/2021.acl-long.297/)
- [BERTweet](https://aclanthology.org/2020.emnlp-main.721/)
- [TGAN: temporal-aware graph attention for early rumor detection](https://research.cuhk.edu.hk/en/publications/tgan-temporal-aware-graph-attention-network-for-early-rumor-detec/)
- [Temporal propagation structure optimization](https://aclanthology.org/2025.coling-main.261.pdf)
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)
- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
- [MAPIE documentation](https://mapie.readthedocs.io/)

