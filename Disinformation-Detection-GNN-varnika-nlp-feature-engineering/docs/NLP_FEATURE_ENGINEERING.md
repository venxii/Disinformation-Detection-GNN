# NLP Feature Engineering (Person 2) — Handoff Document

**Project:** Disinformation Detection Using Graph Neural Networks (PHEME dataset)
**Component:** Person 2 — NLP & Text Representation
**Status:** Stages 1–7 complete. Stage 7 validation result: `PASSED_WITH_WARNINGS` (no failed checks; the warnings are known data and model limitations, listed in [Section J](#j-known-limitations-and-warnings)).
**Primary reader:** Person 3 (GCN/GAT graph model and text-feature integration)

> **Read this first if you only have five minutes**
>
> 1. Cloning the GitHub repository does **not** give you the BERTweet embedding matrix. `embeddings.npy` (104,582 × 768 × 4 bytes ≈ 321 MB, shown as 306M by `du -h`) and `ids.csv` are git-ignored. Get them from Person 2 separately ([Section 4](#4-what-github-provides-and-what-you-must-receive-separately)).
> 2. Use `get_graph_node_embeddings()` from `nlp/embedding_node_helper.py`. It returns `node_ids` and `x`, where `x[i]` is the embedding of `graph["nodes"][i]`.
> 3. Build edges **only** from `graph["edges"]` (`source`/`target` are exact string IDs). **Never** use `parent_id`: it is stored as a JSON number and loses precision.

---

## Contents

- [A. Project overview](#a-project-overview)
- [B. Person 2's completed work (Stages 1–7)](#b-person-2s-completed-work-stages-17)
- [C. Text preprocessing](#c-text-preprocessing)
- [D. TF-IDF feature module](#d-tf-idf-feature-module)
- [E. BERTweet embeddings](#e-bertweet-embeddings)
- [F. Embedding-to-node mapping](#f-embedding-to-node-mapping)
- [G. Exact integration instructions for Person 3](#g-exact-integration-instructions-for-person-3)
- [H. File and directory guide](#h-file-and-directory-guide)
- [I. How to run and validate](#i-how-to-run-and-validate)
- [J. Known limitations and warnings](#j-known-limitations-and-warnings)
- [K. Work not yet completed](#k-work-not-yet-completed)
- [L. Handoff checklist for Person 3](#l-handoff-checklist-for-person-3)
- [4. What GitHub provides and what you must receive separately](#4-what-github-provides-and-what-you-must-receive-separately)

---

## A. Project overview

### What the project does

The project classifies PHEME Twitter conversation threads into four veracity categories: **non-rumour**, **true**, **false**, and **unverified**. Each thread is a source tweet plus its replies. The model will combine:

- **Text**: what each tweet says (Person 2).
- **Propagation structure**: who replied to whom, as a graph (Person 1).
- **Temporal features**: how fast the conversation spread (Person 1).

A graph neural network (Person 3) learns from the text features attached to each node and the reply structure between nodes.

### Dataset statistics (from the Stage 7 report)

| Quantity | Value | Source |
| --- | ---: | --- |
| Conversation threads (graphs) | 6,425 | `data/nlp/stage7_validation_report.json` |
| Posts (tweets) | 104,582 | same |
| BERTweet embeddings | 104,582 | same |
| Graph nodes (over all graphs) | 104,582 | same |
| Edges in graph JSON files | 95,073 | same |
| Rows in `data/processed/edges.csv` | 95,074 | same (1 invalid row; see [Section J](#j-known-limitations-and-warnings)) |
| Single-node graphs | 677 | same |
| Graphs with nodes but no edges | 684 | same |
| Thread labels | non-rumour 4,022 · true 1,067 · unverified 697 · false 638 · unknown 1 | `data/graphs/graph_metadata.csv` |

### What Person 2's component is for

A GNN needs a numeric **feature vector** for every node. Person 2 turns each tweet's text into such a vector:

1. Clean the raw tweet text in two ways (one for BERTweet, one for TF-IDF).
2. Run every cleaned tweet through a frozen BERTweet model and save one 768-number vector per tweet.
3. Provide a helper that, given a graph, returns the vectors in exactly the order of that graph's nodes.

### How the components connect

```
Person 1                         Person 2                                 Person 3
--------                         --------                                 --------
data/processed/posts.csv  --->   nlp/build_clean_text.py
                                   -> data/nlp/posts_clean.csv
                                 nlp/bertweet_embeddings.py
                                   -> data/nlp/embeddings/full/
                                        embeddings.npy  (104582 x 768)
                                        ids.csv         (row -> post_id)
data/graphs/graph_*.json  --->   nlp/embedding_node_helper.py      --->   x          (num_nodes x 768)
  nodes[i]["post_id"]              get_graph_node_embeddings()            edge_index (2 x num_edges)
  edges[k]["source"/"target"]                                             -> GCN / GAT
```

The link between all three components is the tweet's **`post_id`**, an 18-digit string. Every post ID in the project is 18 digits (`post_id_length_counts: {"18": 104582}` in the Stage 7 report).

---

## B. Person 2's completed work (Stages 1–7)

| Stage | Name | Status |
| --- | --- | --- |
| 1 | Environment setup | Complete |
| 2 | Text preprocessing | Complete |
| 3 | TF-IDF feature module | Complete (module and tests only; no official features) |
| 4 | BERTweet pilot | Complete |
| 5 | Full BERTweet extraction | Complete |
| 6 | Embedding-to-graph-node helper | Complete |
| 7 | End-to-end feature validation | Complete (`PASSED_WITH_WARNINGS`) |
| 8 | Text-only baseline training | **Not started** — waiting for Person 4's official split and protocol |
| — | Stance classification | **Not implemented** |

### Stage 1 — Environment setup

- **Purpose:** A reproducible Python environment for the NLP work.
- **Implementation:** A virtual environment at `.venv/` (Python 3.13.5) with pinned packages in `requirements-nlp.txt`. `nlp/check_environment.py` imports each package, prints its version, runs a small tensor on the selected device (`mps` on Apple Silicon, otherwise `cpu`), and checks that BERTweet's tokenizer dependencies import.
- **Design decisions:** Exact version pins. Model downloads go to a project-local cache (`.cache/huggingface/`, from `HF_CACHE_DIR` in `nlp/config.py`), which is git-ignored with `.venv/`.
- **Files created:** `requirements-nlp.txt`, `nlp/check_environment.py`, `nlp/config.py`, `nlp/__init__.py`.
- **Pinned versions:** `numpy==2.5.3`, `pandas==3.0.6`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `torch==2.14.1`, `transformers==5.18.0`, `emoji==2.16.0`, `tqdm==4.70.1`, `pytest==9.1.1`.
- **Validation:** Stage 7 confirmed the installed versions match `requirements-nlp.txt`.

### Stage 2 — Text preprocessing

- **Purpose:** Produce two cleaned text versions of every post, without changing Person 1's `posts.csv`.
- **Implementation:** `nlp/text_preprocessing.py` contains pure, deterministic functions: `basic_clean`, `normalize_for_bertweet`, `normalize_for_tfidf`, and `text_flags`. `nlp/build_clean_text.py` reads `posts.csv` with every column as a string, applies them, and writes `data/nlp/posts_clean.csv`.
- **Design decisions:** IDs are read with `dtype=str, keep_default_na=False`, so 18-digit IDs are never converted to floats and tweets such as `"NA"` stay text. The SHA-256 of `posts.csv` is recorded before and after the build. `PREPROCESSING_VERSION = "1.1"`.
- **Files created:** `nlp/text_preprocessing.py`, `nlp/build_clean_text.py`, `nlp/tests/test_text_preprocessing.py`.
- **Outputs:** `data/nlp/posts_clean.csv` (git-ignored, about 34 MB) and `data/nlp/posts_clean_info.json` (tracked).
- **Validation (`posts_clean_info.json`):** 104,582 posts; row count, exact `post_id`, `root_id`, and original text all verified; `posts.csv unchanged: true`. Summary: 0 empty texts, 2,242 mentions/URL-only posts, 1 empty TF-IDF text, 17,329 posts with URLs, 98,888 with mentions, 20,450 with hashtags, 2,602 with emojis, 1,003 starting with `RT`.

### Stage 3 — TF-IDF feature module

- **Purpose:** A leakage-safe bag-of-words feature extractor for a text-only baseline.
- **Implementation:** `nlp/tfidf_features.py` wraps scikit-learn's `TfidfVectorizer` with project defaults. The vectorizer is fit on a caller-supplied training subset; other subsets are only transformed. A command-line interface takes split CSV files.
- **Design decisions:** Unigrams and bigrams, no stop-word removal, `min_df=2`, sublinear term frequency, L2 normalisation, `float32`. Unknown, duplicate, and overlapping train/test IDs raise errors. Details are in [Section D](#d-tf-idf-feature-module).
- **Files created:** `nlp/tfidf_features.py`, `nlp/tests/test_tfidf_features.py`.
- **Outputs:** None official. Only synthetic test data has been vectorised. `data/nlp/tfidf/` does not exist yet.

### Stage 4 — BERTweet pilot

- **Purpose:** Check that BERTweet loads, tokenises cleaned text correctly, and gives stable embeddings before running all 104,582 posts.
- **Implementation:** `nlp/bertweet_embeddings.py pilot` embeds a deterministic sample of 100 posts (seed 42). `nlp/check_bertweet_tokenizer.py` compares the official BERTweet normalisation with the project's `text_bertweet` column for every post.
- **Pilot results (`data/nlp/embeddings/pilot/metadata.json`):** all 15 pilot checks are `true`, including padding excluded (max difference 4.5e-6), repeat run identical, all parameters frozen, weights unchanged (SHA-256), and MPS vs CPU agreement (max difference 1.96e-5). Token lengths were 4 to 56, with no truncation.
- **Tokenizer check (`data/nlp/bertweet_tokenizer_check.json`):** 104,477 of 104,582 posts normalise identically to the official route. The 105 differences are mostly multi-codepoint emoji such as flags, which the official rule splits into letters (for example `🇫🇷` becomes `:France:` here and `🇫 🇷` officially). There are 0 tokenisation-only differences. Token lengths run from 3 to 180, 99th percentile 38. Two posts exceed 128 tokens. 816 posts contain at least one unknown-token piece, mostly non-Latin scripts.
- **Files created:** `nlp/bertweet_embeddings.py`, `nlp/check_bertweet_tokenizer.py`, `nlp/tests/test_bertweet_embeddings.py`.
- **Outputs:** `data/nlp/embeddings/pilot/` (`metadata.json` tracked; `embeddings.npy` and `ids.csv` ignored) and `data/nlp/bertweet_tokenizer_check.json`.

### Stage 5 — Full BERTweet extraction

- **Purpose:** One frozen BERTweet embedding for every post.
- **Implementation:** `nlp/bertweet_embeddings.py full` processes the posts in 1,024-post chunks. Each chunk is checkpointed, so an interrupted run resumes instead of starting over. Details are in [Section E](#e-bertweet-embeddings).
- **Results (`data/nlp/embeddings/full/metadata.json`):** `status: "complete"`, 104,582 posts, shape `(104582, 768)`, `float32`, MPS (`mps:0`), 630.1 seconds. 103 chunks: 20 reused from an earlier interrupted run, 83 computed. Every recorded check is `true`: shape, dtype, no NaN/Inf, one `ids.csv` row per post, no missing or duplicate `post_id`, IDs match `posts_clean.csv`, a 64-row re-embedded spot check (max difference 0.0), every post processed once, no skipped chunks, weights frozen and unchanged, evaluation mode, and a manual mean-pooling check (max difference 1.4e-6).
- **Files created:** `nlp/tests/test_bertweet_checkpoints.py`. Extraction code is in `nlp/bertweet_embeddings.py`.
- **Outputs:** `data/nlp/embeddings/full/embeddings.npy`, `ids.csv` (both git-ignored), and `metadata.json` (tracked).
- **Tests:** 132 passed at the end of Stage 5.

### Stage 6 — Embedding-to-graph-node helper

- **Purpose:** Give Person 3 a safe way to get node features in graph order.
- **Implementation:** `nlp/embedding_node_helper.py` (API in [Section F](#f-embedding-to-node-mapping)). `nlp/check_node_helper.py` checks it against Person 1's graphs.
- **Results (`data/nlp/node_helper_check.json`):** `passed: true`; memory-mapped store. All 6,425 graphs: 104,582 nodes, 104,582 distinct post IDs, 0 graphs with problems, 0 IDs in more than one graph, 0 unused embedding rows. Six graphs (the largest with 346 nodes, a single-node graph, a three-node graph with no edges, and three random graphs) were re-embedded from their own `text` field and matched the helper rows within 1e-3 (max difference about 1e-6). Shuffling a graph's node list shuffled the returned rows the same way.
- **Files created:** `nlp/embedding_node_helper.py`, `nlp/check_node_helper.py`, `nlp/tests/test_embedding_node_helper.py`.
- **Outputs:** `data/nlp/node_helper_check.json`.
- **Tests:** 165 passed at the end of Stage 6.

### Stage 7 — End-to-end feature validation

- **Purpose:** Check the whole chain `posts.csv` → `posts_clean.csv` → BERTweet `ids.csv` → graph nodes, without regenerating anything.
- **Implementation:** `nlp/stage7_validation.py` runs 49 checks:
  - Cleaned text was compared with the originals by `post_id`, and both normalisers were rerun on every post.
  - TF-IDF guarantees were tested on synthetic posts.
  - The embedding store was checked in memory-mapped mode, including a NaN/Inf scan.
  - Every graph was walked: node order, the embedding row for each node, edge-index validity, and `root_id` membership.
  - Thread IDs were compared across posts, claims, and graph metadata.
  - Repository safety: Person 1 files unchanged, ignore rules, no credentials or absolute paths, package pins, and file hashes before and after.
  - The full test suite was run.
- **Result:** `PASSED_WITH_WARNINGS`; 49 of 49 checks passed. The two posts, cleaned-posts, embedding-ID, and graph-node ID sets all contain the same 104,582 IDs (0 missing, 0 extra). Stage 6's re-embedding check was reused, not rerun.
- **Files created:** `nlp/stage7_validation.py`, `nlp/tests/test_stage7_validation.py`.
- **Outputs:** `data/nlp/stage7_validation_report.json`, `data/nlp/STAGE7_VALIDATION_REPORT.md`.
- **Tests:** **180 passed, 0 failed, 0 skipped.**

---

## C. Text preprocessing

### Three text columns

| Column | What it is | Used by |
| --- | --- | --- |
| `text_original` | The `text` field of `posts.csv`, byte-for-byte. | Reference and validation |
| `text_bertweet` | Text normalised the way BERTweet's pre-training data was. Case, punctuation, hashtags, and `RT` are kept. | BERTweet embeddings |
| `text_tfidf` | Lower-cased, space-separated tokens with placeholder words for URLs, mentions, emojis, emoticons, `?`, and `!`. | TF-IDF |

### Why two cleaned versions?

The two models need opposite things. BERTweet was pre-trained on tweets normalised in a specific way (`@USER`, `HTTPURL`, emoji names, original casing), and it understands punctuation and case. TF-IDF only counts words, so it needs consistent lower-case tokens, and important signals such as `?`, `!`, and negation must be explicit words so they are not discarded as punctuation.

### Shared cleaning (`basic_clean`)

Both versions start with:

1. Missing values become `""`.
2. HTML entities are decoded (`&amp;` → `&`, `&gt;` → `>`, `&lt;` → `<`).
3. Newlines, tabs, and repeated spaces collapse to one space, and the ends are trimmed.

### `normalize_for_bertweet`

This follows the official BERTweet normalisation (`BertweetTokenizer.normalizeTweet`) using the same `TweetTokenizer`:

- `@mentions` → `@USER`; tokens starting with `http` or `www` → `HTTPURL`.
- Emojis → text names via `emoji.demojize` (for example 😢 → `:crying_face:`).
- `’` → `'` and `…` → `...`.
- Contractions are split as in pre-training (`don't` → `do n't`). Because the official rule then rejoins `ca n't` as `can't`, the result is `can't`.
- Hashtags, capital letters, punctuation, emoticons, and `RT` are kept.
- **Addition not in the official code:** multi-codepoint emojis (flags, ❤️, ZWJ sequences) are converted to names before tokenisation. The official tokenizer splits them into pieces that cannot be converted afterwards.

### `normalize_for_tfidf`

1. Shared cleaning.
2. URLs → `xxurl`, `@mentions` → `xxuser`. Email addresses are not treated as mentions.
3. Emojis → one token each (😢 → `xxemoji_crying_face`). Standalone emoticons → tokens such as `xxemoticon_smile`, `xxemoticon_sad`, `xxemoticon_heart`, `xxemoticon_wink`, `xxemoticon_laugh`, `xxemoticon_tongue`, `xxemoticon_skeptical`, `xxemoticon_cry`, and `xxemoticon_brokenheart`. Only standalone emoticons are converted, so `10:30` is not read as a face.
4. Lower-casing.
5. Contractions are expanded and negations are kept as `not`. This covers `can't`, `won't`, `shan't`, `ain't`, `cannot`, apostrophe-free forms such as `dont` and `isnt`, and the general `n't` rule. `'re`, `'m`, `'ll`, and `'ve` are expanded. The ambiguous `'s` and `'d` are dropped.
6. Runs of `?` or `？` → `xxqmark`; runs of `!` or `！` → `xxexcl`.
7. All other punctuation, including the `#` of hashtags, becomes a space. The hashtag word is kept (`#CharlieHebdo` → `charliehebdo`).
8. No stop words are removed, and non-English letters are kept.

### Preprocessing flags (`text_flags`)

Computed on the shared-cleaned original text:

| Flag | Meaning |
| --- | --- |
| `num_urls`, `num_mentions`, `num_hashtags`, `num_emojis` | Counts in the cleaned original text |
| `starts_with_rt` | Text begins with `RT` as a word |
| `is_empty_text` | Nothing left after shared cleaning |
| `only_mentions_urls` | Not empty, but nothing remains once mentions and URLs are removed (`"@user ?"` counts as content) |

In the CSV, booleans are stored as the strings `"True"` and `"False"` because the file is read with `dtype=str`.

### Example (actual output of the current implementation)

Input:

```text
RT @BBCNews: Police can't confirm 12 dead &amp; gunman at large?! 😢 #CharlieHebdo http://t.co/abc :(
```

`normalize_for_bertweet`:

```text
RT @USER : Police can't confirm 12 dead & gunman at large ? ! :crying_face: #CharlieHebdo HTTPURL :(
```

`normalize_for_tfidf`:

```text
rt xxuser police can not confirm 12 dead gunman at large xxqmark xxexcl xxemoji_crying_face charliehebdo xxurl xxemoticon_sad
```

`text_flags`:

```python
{'num_urls': 1, 'num_mentions': 1, 'num_hashtags': 1, 'num_emojis': 1,
 'starts_with_rt': True, 'is_empty_text': False, 'only_mentions_urls': False}
```

---

## D. TF-IDF feature module

**Module:** `nlp/tfidf_features.py`

### What TF-IDF is

TF-IDF turns each post into a long, mostly zero vector with one column per vocabulary term. A term's weight is higher when it appears often in this post (term frequency) and lower when it appears in many posts (inverse document frequency). It is a simple, interpretable text baseline.

### Settings (`DEFAULT_TFIDF_PARAMS`)

| Parameter | Value | Why |
| --- | --- | --- |
| `ngram_range` | `(1, 2)` | Unigrams plus bigrams, so phrases such as `not true` are kept |
| `token_pattern` | `r"(?u)\S+"` | `text_tfidf` is already tokenised, so a token is anything between spaces. This keeps placeholders, `not`, and single characters. |
| `lowercase` | `False` | Already lower-cased in Stage 2 |
| `stop_words` | `None` | Words such as `not` and `is` carry signal for rumours |
| `strip_accents` | `None` | Non-English text is kept unchanged |
| `min_df` | `2` | Terms must appear in at least 2 training posts |
| `max_df`, `max_features` | `1.0`, `None` | No upper cut-off or vocabulary cap |
| `sublinear_tf` | `True` | Uses `1 + log(tf)`, so repeated words do not dominate |
| `norm` | `"l2"` | Each row has unit length |
| `dtype` | `np.float32` | Smaller matrices |

### Train-only fitting (no leakage)

- `fit_tfidf(train_subset)` learns the vocabulary and IDF weights **only** from the training posts.
- `transform_tfidf(vectorizer, subset)` applies the already-fitted vectorizer to validation and test posts. Words never seen in training are ignored.
- `check_no_overlap(train_subset, other_subset, name)` raises an error if a post appears in both.

Stage 7 confirmed on synthetic data that transforming held-out posts leaves `vocabulary_` and `idf_` unchanged.

### Public functions

| Function | Purpose |
| --- | --- |
| `load_clean_posts(path=POSTS_CLEAN_FILE)` | Load `posts_clean.csv` with every column as a string |
| `select_posts(posts, post_ids=None, root_ids=None, post_type="all")` | Select rows by post IDs (requested order is kept) or by thread `root_id`s. `post_type` is `"all"`, `"source"`, or `"reaction"`. Unknown or duplicate IDs raise `ValueError`. |
| `make_vectorizer(**overrides)` | `TfidfVectorizer` with project defaults |
| `fit_tfidf(train_subset, **overrides)` | Fit on training posts; returns `(vectorizer, csr_matrix)` |
| `transform_tfidf(vectorizer, subset)` | Transform other posts |
| `check_no_overlap(train_subset, other_subset, name)` | Reject train/other post overlap |
| `save_features(output_dir, name, matrix, subset)` / `load_features(output_dir, name)` | `<name>_matrix.npz` (sparse) plus `<name>_ids.csv` (`row_index`, `post_id`, `root_id`) |
| `save_vectorizer(vectorizer, output_dir, train_subset, extra_info=None)` / `load_vectorizer(output_dir)` | `vectorizer.joblib` plus a readable `vectorizer_info.json` |

### Usage once the official split exists (do not run before then)

```python
from nlp.tfidf_features import (
    load_clean_posts, select_posts, fit_tfidf, transform_tfidf, check_no_overlap
)

posts = load_clean_posts()
train = select_posts(posts, root_ids=train_root_ids)  # from Person 4's split
test = select_posts(posts, root_ids=test_root_ids)
check_no_overlap(train, test, "test")

vectorizer, X_train = fit_tfidf(train)
X_test = transform_tfidf(vectorizer, test)
# Row i of X_train belongs to train["post_id"].iloc[i]
```

Command-line form (writes to `data/nlp/tfidf/<run-name>/` and refuses to overwrite an existing run unless `--overwrite` is passed):

```bash
.venv/bin/python -m nlp.tfidf_features \
    --run-name <run-name> \
    --train-split <train.csv> \
    --transform-split val=<val.csv> \
    --transform-split test=<test.csv>
```

Split CSV files need a `root_id` column (whole threads) or a `post_id` column. Splitting by `root_id` keeps all posts of a conversation in the same split.

> **Not done yet:** No official TF-IDF matrices exist, and no TF-IDF baseline has been trained. Person 4's official conversation-level split is needed first.

---

## E. BERTweet embeddings

### Model

| Setting | Value |
| --- | --- |
| Model | `vinai/bertweet-base` |
| Pinned revision | `b349c1243407b0dcffeabb2337497477286e27ab` |
| Weights | safetensors (recorded as identical to `pytorch_model.bin`) |
| Input column | `text_bertweet` (tokenizer created with `normalization=False`, because the text is already normalised) |
| Max sequence length | 128 tokens, `truncation=True` |
| Output | 768-dimensional `float32` vector per post |

**Why BERTweet?** It is a RoBERTa-base model pre-trained on English tweets, so it handles Twitter-specific text (mentions, URLs, hashtags, emoji names, informal spelling) better than a model trained only on formal text.

### How a vector is produced (`embed_texts` and `mean_pool`)

1. **Tokenisation:** The tokenizer splits text into sub-word pieces and adds `<s>` and `</s>`. Texts longer than 128 tokens are cut off.
2. **Padding and attention mask:** Texts in a batch are padded to the same length. The attention mask is 1 for real tokens and 0 for padding.
3. **Forward pass:** The model returns one 768-number vector per token from its last hidden layer.
4. **Mean pooling:** Token vectors are averaged, multiplying by the mask first so padding contributes nothing. `<s>` and `</s>` are included. A row with no real tokens returns zeros instead of dividing by zero.

```python
mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
summed = (last_hidden_state * mask).sum(dim=1)
counts = mask.sum(dim=1).clamp(min=1.0)
return summed / counts
```

### Frozen weights (no fine-tuning)

`freeze_model()` puts the model in evaluation mode and sets `requires_grad=False` on every parameter. Every forward pass runs under `torch.inference_mode()`. The full run recorded a parameter fingerprint before and after extraction (`"11. model weights frozen and unchanged": true`).

### Device

Extraction ran on Apple Silicon MPS (`requested_device: "mps"`, `actual_device: "mps:0"`, no CPU fallback). The pilot confirmed MPS and CPU agree within 2e-5. You do **not** need MPS or a GPU to *load* the saved embeddings.

### Checkpoint and resume

The full run splits posts into 1,024-post chunks and writes each chunk atomically (temporary file, then rename) under `data/nlp/checkpoints/bertweet_full/` (git-ignored). On restart, valid chunks are reused. A chunk with the wrong IDs or contents is recomputed. If the input file hash, post IDs, or settings changed, the run refuses to reuse checkpoints. The final output is written only after all chunks are assembled and checked. `run_full` refuses to overwrite a completed extraction unless `--overwrite` is passed. Do not pass it.

### Artifacts

| Path | Content | In Git? |
| --- | --- | --- |
| `data/nlp/embeddings/full/embeddings.npy` | `(104582, 768)` `float32` matrix, about 321 MB | **No** (ignored) |
| `data/nlp/embeddings/full/ids.csv` | Columns `embedding_index`, `post_id`, `root_id`; 104,582 rows; `embedding_index` is 0…104581 | **No** (ignored) |
| `data/nlp/embeddings/full/metadata.json` | Status, model, revision, pooling, settings, hashes, checks, warnings | Yes (ignore-rule exception) |

`metadata.json` also records `input_sha256` (the hash of the `posts_clean.csv` used) and `post_ids_sha256`. Stage 7 confirmed the current `posts_clean.csv` still matches `input_sha256`.

---

## F. Embedding-to-node mapping

**This is the section Person 3 needs most.**

### Why row order cannot be trusted

`embeddings.npy` stores posts in `posts_clean.csv` order. Each graph JSON lists its nodes in its own order. They are unrelated, and the embedding file contains all 104,582 posts while a graph contains only its own thread. If you take embedding rows `0..n-1` for a graph with `n` nodes, every node gets someone else's vector, and nothing raises an error.

The only safe link is the exact string `post_id`:

```
graph["nodes"][i]["post_id"]  ->  ids.csv: post_id -> embedding_index  ->  embeddings.npy[embedding_index]
```

### Graph JSON shape (Person 1's files, read-only)

```json
{
  "root_id": "498235547685756928",
  "label": "non-rumour",
  "num_nodes": 14,
  "num_edges": 13,
  "nodes": [
    {"post_id": "498235547685756928", "text": "...", "timestamp": "...", "parent_id": null, "post_type": "source"},
    {"post_id": "498243332204949504", "text": "...", "timestamp": "...", "parent_id": 4.982355476857569e+17, "post_type": "reaction"}
  ],
  "edges": [
    {"source": "498235547685756928", "target": "498243332204949504"}
  ],
  "temporal_features": {"...": "..."}
}
```

- A node's index is its **position** in `nodes`. There is no separate numeric node ID.
- `post_id`, `edges[k]["source"]`, and `edges[k]["target"]` are **strings**.
- `source` is the parent post and `target` is the reply (from `dataset/graph_export.py`).
- `parent_id` is a JSON **number** (float). **Do not use it.** See the warning below.

### `nlp/embedding_node_helper.py` API

| Function / class | What it does |
| --- | --- |
| `load_embedding_store(output_dir=FULL_EMBEDDINGS_DIR, mmap=True)` | Loads the full embeddings through `load_full_embeddings()`. It refuses to load unless `metadata.json` has `status: "complete"`, the shape matches the metadata, `embedding_index` is `0..n-1`, and there are no duplicate IDs. Returns an `EmbeddingStore`. With `mmap=True` the matrix stays on disk and only requested rows are read. |
| `EmbeddingStore` | Frozen dataclass: `embeddings` (array or memmap), `index` (`dict` from `post_id` to row), `metadata` (dict). Also has `dim`, `len(store)`, and `post_id in store`. |
| `get_node_embeddings(node_post_ids, store)` | Returns a new in-memory `float32` array of shape `(len(node_post_ids), 768)` with row `i` = embedding of `node_post_ids[i]`. Input order is kept exactly. An empty list gives shape `(0, 768)`. |
| `get_graph_node_embeddings(graph, store)` | `graph` is a dict or a path to a graph JSON file. Returns `(node_post_ids, features)` where `features[i]` belongs to `graph["nodes"][i]`. |
| `graph_node_post_ids(graph)` | The `post_id` of every node, in `nodes` order |
| `load_graph(path)` | Reads one graph JSON (read-only) |
| `build_embedding_store(embeddings, post_ids, metadata=None)` | Builds a store from any matrix plus IDs, for example in tests |

### Error handling

Nothing is silently skipped, and no node gets another node's vector:

| Situation | Exception |
| --- | --- |
| A `post_id` is not a non-empty string without surrounding spaces (for example an `int`, `float`, `None`, or `""`) | `InvalidPostIdError` |
| A single string is passed instead of a list | `InvalidPostIdError` |
| The same `post_id` is requested twice, or appears twice in the ID mapping | `DuplicatePostIdError` |
| A `post_id` has no embedding (including a float-rounded ID) | `MissingPostIdError` (also a `KeyError`) |
| Matrix rows ≠ number of IDs, or the matrix is not 2-D | `MappingMismatchError` |
| A graph has no `nodes` list, or a node has no `post_id` | `InvalidPostIdError` |

All of these subclass `EmbeddingLookupError`.

### Working integration example

This code was run against the real store and graphs while writing this document:

```python
import numpy as np
from nlp.embedding_node_helper import (
    load_embedding_store, load_graph, get_graph_node_embeddings
)

store = load_embedding_store()              # once; memory-mapped
graph = load_graph("data/graphs/graph_000001.json")

node_ids, x = get_graph_node_embeddings(graph, store)
# x.shape == (14, 768), x.dtype == float32
# x[i] is the embedding of node_ids[i] == graph["nodes"][i]["post_id"]

position = {post_id: i for i, post_id in enumerate(node_ids)}

edge_index = np.array(
    [[position[e["source"]] for e in graph["edges"]],
     [position[e["target"]] for e in graph["edges"]]],
    dtype=np.int64,
).reshape(2, -1)
# graph_000001.json -> edge_index.shape == (2, 13)
# a graph with no edges -> edge_index.shape == (2, 0)
```

`position` is built from the **same `node_ids` list** that ordered `x`, so edge index `j` refers to row `x[j]`. A `KeyError` from `position[...]` means an edge points to a post that is not a node. Stage 7 found no such edge in any graph.

> **Critical warning: do not reconstruct connectivity from `parent_id`.**
> `parent_id` was written as a JSON floating-point number. A float stores only about 15–17 significant digits, while Twitter IDs have 18. In Stage 7, 32,782 of the 104,582 post IDs changed when passed through a float, and 29,369 of the 95,073 edges disagreed with the parent obtained by converting `parent_id` back to text. Use `graph["edges"]` and its exact string `source` and `target` values only.

---

## G. Exact integration instructions for Person 3

The GNN framework has not been chosen in this repository yet. No Person 3 model code exists, and `requirements.txt` is empty. The steps below give NumPy arrays, then show how to convert them to PyTorch. `torch` is already pinned in `requirements-nlp.txt`.

### Step 1 — Set up the environment

From the project root (Python 3.13 was used):

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-nlp.txt
.venv/bin/python -m nlp.check_environment
```

Always run project code from the project root with `.venv/bin/python` so the `nlp` package imports correctly.

### Step 2 — Make sure the artifacts are present

Place the files you received from Person 2 here ([Section 4](#4-what-github-provides-and-what-you-must-receive-separately)):

```text
data/nlp/embeddings/full/embeddings.npy
data/nlp/embeddings/full/ids.csv
data/nlp/embeddings/full/metadata.json
data/graphs/graph_*.json            (from Person 1; tracked in Git)
```

Quick check:

```bash
.venv/bin/python -c "from nlp.embedding_node_helper import load_embedding_store as l; s=l(); print(len(s), s.dim, type(s.embeddings).__name__)"
```

Expected output: `104582 768 memmap`.

### Step 3 — Load the embedding store once

```python
from nlp.embedding_node_helper import load_embedding_store
store = load_embedding_store()   # reuse for every graph; do not reload per graph
```

### Step 4 — Load a graph

```python
from nlp.embedding_node_helper import load_graph
graph = load_graph("data/graphs/graph_000001.json")
```

### Step 5 — Get `node_ids` and `x`

```python
from nlp.embedding_node_helper import get_graph_node_embeddings
node_ids, x = get_graph_node_embeddings(graph, store)   # x: (num_nodes, 768) float32
```

### Step 6 — Build the node-position dictionary

```python
position = {post_id: i for i, post_id in enumerate(node_ids)}
```

### Step 7 — Convert edge endpoints to integer indices

```python
import numpy as np
edge_index = np.array(
    [[position[e["source"]] for e in graph["edges"]],
     [position[e["target"]] for e in graph["edges"]]],
    dtype=np.int64,
).reshape(2, -1)
```

Edges point from parent to reply. Whether to add reverse edges or self-loops is a GCN/GAT design choice for Person 3.

### Step 8 — Convert to tensors

```python
import torch
x_t = torch.from_numpy(x)                    # float32, (num_nodes, 768)
edge_index_t = torch.from_numpy(edge_index)  # int64, (2, num_edges)
```

If the team adopts PyTorch Geometric (not installed or confirmed in this repository), these tensors match its `Data(x=..., edge_index=...)` convention. The thread label is available as `graph["label"]`.

### The one rule

**`x[i]` must always correspond to graph node `i`.** Do not sort, filter, or reorder `x` without applying the same change to `node_ids` and rebuilding `position` and `edge_index`. If you need only some nodes, pass them in the order you want to `get_node_embeddings()`. The rows come back in that order.

### Notes

- **Graph sizes:** 677 graphs have a single node, and 684 graphs have nodes but no edges (`edge_index` shape `(2, 0)`). There are no empty graphs. The helper handles all of these.
- **Labels:** One graph has label `unknown` ([Section J](#j-known-limitations-and-warnings)). Decide with Person 4 how to treat it.
- **Memory:** With `mmap=True`, loading the store and walking all 6,425 graphs peaked at about 727 MB of process memory in Stage 6. The returned `x` is an in-memory copy, so writing to it does not change the store.

---

## H. File and directory guide

### Source code, tests, configuration, and small metadata (tracked in Git once committed)

| Path | Type | Purpose |
| --- | --- | --- |
| `nlp/__init__.py` | Source | Package marker |
| `nlp/config.py` | Config | All paths, `PREPROCESSING_VERSION`, model name, pinned revision, `BERTWEET_MAX_LENGTH = 128`, `EMBEDDING_DIM = 768` |
| `nlp/check_environment.py` | Script | Package, device, and tokenizer-support check |
| `nlp/text_preprocessing.py` | Source | `basic_clean`, `normalize_for_bertweet`, `normalize_for_tfidf`, `text_flags` |
| `nlp/build_clean_text.py` | Script | Builds `posts_clean.csv` from `posts.csv` |
| `nlp/tfidf_features.py` | Source / CLI | Train-only TF-IDF fit/transform, save/load |
| `nlp/bertweet_embeddings.py` | Source / CLI | Model loading, mean pooling, pilot and checkpointed full extraction, `load_full_embeddings` |
| `nlp/check_bertweet_tokenizer.py` | Script | Official vs project BERTweet normalisation comparison |
| `nlp/embedding_node_helper.py` | Source | **Integration API for Person 3** |
| `nlp/check_node_helper.py` | Script | Stage 6 integration check (loads BERTweet) |
| `nlp/stage7_validation.py` | Script | Stage 7 end-to-end validation |
| `nlp/tests/test_text_preprocessing.py` | Tests | 55 tests |
| `nlp/tests/test_tfidf_features.py` | Tests | 26 tests |
| `nlp/tests/test_bertweet_embeddings.py` | Tests | 24 tests (fake tokenizer/model; no download) |
| `nlp/tests/test_bertweet_checkpoints.py` | Tests | 27 tests (checkpoint/resume) |
| `nlp/tests/test_embedding_node_helper.py` | Tests | 33 tests (synthetic matrices) |
| `nlp/tests/test_stage7_validation.py` | Tests | 15 tests |
| `requirements-nlp.txt` | Config | Pinned NLP dependencies |
| `.gitignore` | Config | Excludes `.venv/`, `.cache/`, large NLP outputs |
| `data/nlp/posts_clean_info.json` | Metadata | Stage 2 summary and verification |
| `data/nlp/bertweet_tokenizer_check.json` | Report | Stage 4 tokenizer comparison |
| `data/nlp/embeddings/pilot/metadata.json` | Metadata | Stage 4 pilot settings and checks |
| `data/nlp/embeddings/full/metadata.json` | Metadata | Stage 5 settings, hashes, checks, warnings |
| `data/nlp/node_helper_check.json` | Report | Stage 6 results |
| `data/nlp/stage7_validation_report.json` | Report | Stage 7 results (machine-readable) |
| `data/nlp/STAGE7_VALIDATION_REPORT.md` | Report | Stage 7 results (readable) |
| `docs/NLP_FEATURE_ENGINEERING.md` | Docs | This document |

### Generated data artifacts (git-ignored; not on GitHub)

| Path | Size | Needed by Person 3? |
| --- | --- | --- |
| `data/nlp/embeddings/full/embeddings.npy` | about 321 MB | **Yes** |
| `data/nlp/embeddings/full/ids.csv` | about 4.4 MB | **Yes** |
| `data/nlp/posts_clean.csv` | about 34 MB | No for node features. Yes for TF-IDF, rebuilding cleaned text, or running Stage 7. |
| `data/nlp/embeddings/pilot/embeddings.npy`, `ids.csv` | small | No |
| `data/nlp/checkpoints/` | — | No (extraction checkpoints) |
| `.cache/huggingface/` | several hundred MB | No, unless re-embedding text |
| `.venv/` | — | No; create your own |

### Person 1's inputs (tracked; read-only for Person 2)

`data/processed/posts.csv`, `claims.csv`, `edges.csv`, `data/graphs/graph_*.json`, `data/graphs/graph_metadata.csv`, `dataset/*.py`, and `docs/data_pipeline.md`.

---

## I. How to run and validate

All commands run from the project root.

**Use the environment** (no activation needed when calling `.venv/bin/python` directly):

```bash
source .venv/bin/activate      # optional; otherwise use .venv/bin/python
.venv/bin/python -m nlp.check_environment
```

**Run the NLP test suite** (Stage 7 result: 180 passed, 0 failed, 0 skipped):

```bash
.venv/bin/python -m pytest nlp/tests -q
```

The tests use synthetic data and fake models. They do not need the embedding files or a model download.

**Check that the embedding store loads:**

```bash
.venv/bin/python -c "from nlp.embedding_node_helper import load_embedding_store as l; s=l(); print(len(s), s.dim, s.embeddings.dtype)"
```

Expected output: `104582 768 float32`.

**Re-run the Stage 7 validation (optional):**

```bash
.venv/bin/python -m nlp.stage7_validation
```

This reads `posts.csv`, `posts_clean.csv`, the embeddings, and all graphs. It takes about a minute. It **rewrites** the two Stage 7 report files but does not modify any data or embeddings.

**Do not run these as normal setup.** They regenerate completed artifacts or need the model download:

- `.venv/bin/python -m nlp.bertweet_embeddings full` (and especially `--overwrite` or `--restart`): re-extracts all embeddings. It refuses to overwrite a completed extraction without `--overwrite`.
- `.venv/bin/python -m nlp.build_clean_text`: rewrites `posts_clean.csv` and `posts_clean_info.json`. If the content changed, the embedding metadata hash would no longer match.
- `.venv/bin/python -m nlp.check_node_helper`: loads BERTweet and rewrites `node_helper_check.json`.

---

## J. Known limitations and warnings

None of these are implementation failures. Stage 7 passed every check. They are data or model properties that downstream work should know about.

### Data and model limitations

| Topic | Details | Impact |
| --- | --- | --- |
| English-focused model | `vinai/bertweet-base` was pre-trained on English tweets. Non-English posts were kept and embedded. 816 posts contain unknown-token pieces, mostly non-Latin scripts. | Weaker representations for non-English posts |
| Truncation | Two posts exceed 128 tokens and were truncated: `524987107346243584`, `525434720390488064` (max token length 180). | Their vectors represent only the first 128 tokens |
| Identical texts | 7,315 posts share a cleaned BERTweet text with at least one other post (1,685 distinct texts). Their embeddings are identical, as expected. | Not an alignment error. Each post still has its own row. |
| Empty TF-IDF text | 1 post has non-empty BERTweet text but empty TF-IDF text (punctuation only). | It will become an all-zero TF-IDF row once a vectorizer is fit |
| `unknown` label | One thread is labelled `unknown`. Per `docs/data_pipeline.md`, its PHEME `is_rumour` value was "unclear". | Outside the four classes; decide with Person 4 how to treat it |
| `parent_id` precision | `parent_id` is a JSON float. 32,782 post IDs change if passed through a float, and 29,369 of 95,073 edges disagree with a float-reconstructed parent. | **Never** build edges from `parent_id`; use `edges` |
| Edge count | `docs/data_pipeline.md` and `edges.csv` report 95,074 edges; the graph files and `graph_metadata.csv` contain 95,073. One `edges.csv` row in thread `521310417696858112` (parent `521310417696858112`) has a malformed `child_id` that begins `521311862118711296”:[],”521313155004497920”:[],…` (66 characters, not a valid ID). `dataset/graph_export.py` skipped it. | Graph files are internally consistent; the documentation count includes the invalid row. Person 1 may want to note this. |

### Implementation note (not a failure)

| Topic | Details |
| --- | --- |
| `load_features` duplicate-ID check | `nlp/tfidf_features.py` `load_features()` checks that the matrix row count matches the ID file and that `row_index` is `0..n-1`, but it does not reject a duplicated `post_id` inside a saved `*_ids.csv`. `select_posts()` and `fit_tfidf()` reject duplicates before anything is saved, so this only matters if a saved ID file is edited by hand. |

---

## K. Work not yet completed

| Item | Status |
| --- | --- |
| Stage 8: text-only baselines (TF-IDF and BERTweet classifiers) | **Not trained** |
| Official TF-IDF features | **Not generated.** Person 4's official split is not available yet. |
| Official model evaluation by Person 2 | **Not performed.** There are no reported accuracy or F1 numbers. |
| BERTweet fine-tuning | **Not done.** The embeddings come from frozen pretrained weights. |
| Stance classification | **Not implemented** |
| GNN implementation or training | **Not done by Person 2** (Person 3's responsibility) |
| Official train/validation/test split | **Not created by Person 2** (Person 4's responsibility) |

**Next dependency:** Person 4 should provide the official **conversation-level** split, as lists of `root_id` per split, so that all posts of a thread stay together. They should also provide the evaluation protocol (metrics, handling of the `unknown` label, and any repeated runs or seeds). Stage 8 text-only baselines will be trained only after that.

---

## 4. What GitHub provides and what you must receive separately

The NLP work is currently **uncommitted** (`nlp/`, `data/nlp/`, and `requirements-nlp.txt` are untracked, and `.gitignore` is modified). Once Person 2 commits and pushes, the `.gitignore` rules below decide what reaches GitHub.

### From GitHub (after Person 2 pushes)

- All `nlp/` source code and `nlp/tests/`
- `requirements-nlp.txt` and `.gitignore`
- `docs/NLP_FEATURE_ENGINEERING.md`
- Small metadata and reports: `data/nlp/posts_clean_info.json`, `data/nlp/bertweet_tokenizer_check.json`, `data/nlp/node_helper_check.json`, `data/nlp/stage7_validation_report.json`, `data/nlp/STAGE7_VALIDATION_REPORT.md`, `data/nlp/embeddings/full/metadata.json`, `data/nlp/embeddings/pilot/metadata.json`
- Person 1's files that are already tracked: `data/processed/*`, `data/graphs/*`, `dataset/*.py`, `docs/data_pipeline.md`

### Not on GitHub — receive these separately from Person 2

| File | Why it is not on GitHub | Required? |
| --- | --- | --- |
| `data/nlp/embeddings/full/embeddings.npy` | about 321 MB; ignored by `data/nlp/embeddings/**` | **Yes** |
| `data/nlp/embeddings/full/ids.csv` | Ignored by the same rule | **Yes.** Without it the matrix rows cannot be matched to posts. |
| `data/nlp/posts_clean.csv` | about 34 MB; ignored | Only for TF-IDF or Stage 7 validation |

`.cache/huggingface/` (model cache) and `.venv/` are also not on GitHub, and you do not need them. Create your own `.venv` from `requirements-nlp.txt`. The model is only needed to re-embed text.

**Transfer note:** These files have not been uploaded to any shared storage yet. The team needs to agree how to transfer them, for example a shared drive. Keep `embeddings.npy`, `ids.csv`, and `metadata.json` from the **same** extraction together. `load_embedding_store()` checks that the matrix shape matches `metadata.json` and that `ids.csv` lines up with the matrix. To confirm the transferred `ids.csv` matches, its post IDs (joined by `\n`) should hash to `post_ids_sha256` in `metadata.json`:

```bash
.venv/bin/python -c "import hashlib,json,pandas as pd; ids=pd.read_csv('data/nlp/embeddings/full/ids.csv',dtype=str)['post_id']; m=json.load(open('data/nlp/embeddings/full/metadata.json')); print(hashlib.sha256('\n'.join(ids).encode()).hexdigest()==m['post_ids_sha256'])"
```

Expected output: `True`.

---

## L. Handoff checklist for Person 3

- [ ] NLP source code available (`nlp/`, pulled from GitHub)
- [ ] Dependencies and environment configured (`.venv` from `requirements-nlp.txt`; `nlp.check_environment` passes)
- [ ] Full BERTweet embedding matrix available locally (`data/nlp/embeddings/full/embeddings.npy`, received separately)
- [ ] Embedding ID mapping available locally (`data/nlp/embeddings/full/ids.csv`, received separately)
- [ ] Embedding metadata available (`data/nlp/embeddings/full/metadata.json`, `status: "complete"`)
- [ ] Person 1 graph files available (`data/graphs/graph_*.json`, 6,425 files)
- [ ] Node helper loads the embedding store (`load_embedding_store()` gives 104,582 rows × 768)
- [ ] Graph features preserve node ordering (`x[i]` ↔ `node_ids[i]` ↔ `graph["nodes"][i]`)
- [ ] Edge indices are built from exact string IDs (`graph["edges"]` `source`/`target` via the `position` dictionary)
- [ ] `parent_id` is not used to construct graph edges
