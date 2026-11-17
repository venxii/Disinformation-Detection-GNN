"""Step 2: event-cue masking, then re-embedding with the exact frozen-BERTweet pipeline.

Masking (applied to the raw tweet text, before BERTweet normalisation):
  URLs      -> HTTPURL       (the original pipeline already did this)
  @mentions -> @USER         (the original pipeline already did this)
  #hashtags -> HASHTAG       (new)
  spaCy en_core_web_lg entities (new):
      PERSON          -> PERSON
      GPE / LOC / FAC -> LOCATION
      ORG             -> ORGANIZATION
Then nlp/text_preprocessing.normalize_for_bertweet and mean-pooled last-layer BERTweet at
the same revision / max_length as data/nlp/embeddings/full (see its metadata.json).

    .venv/bin/python -m experiments.generalization.masking --check 500     # reproduce unmasked rows
    .venv/bin/python -m experiments.generalization.masking                 # write data/nlp/embeddings/masked

Rows follow data/nlp/embeddings/full/ids.csv, so lifecycle.data.load_threads(embedding_dir=...)
works unchanged. Masking is per-post and label-free, so it cannot leak across folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NLP_ROOT = Path(__file__).resolve().parents[2] / "Disinformation-Detection-GNN-varnika-nlp-feature-engineering"
sys.path.insert(0, str(NLP_ROOT))
from nlp.text_preprocessing import HASHTAG_RE, MENTION_RE, URL_RE, basic_clean, normalize_for_bertweet  # noqa: E402

FULL = Path("data/nlp/embeddings/full")
REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"
ENTITY_MAP = {"PERSON": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION", "ORG": "ORGANIZATION"}
PLACEHOLDERS = {"HTTPURL", "@USER", "HASHTAG", "USER"}


def pre_mask(text: str) -> str:
    text = basic_clean(text)
    text = URL_RE.sub(" HTTPURL ", text)
    text = MENTION_RE.sub(" @USER ", text)
    return re.sub(r"\s+", " ", HASHTAG_RE.sub(" HASHTAG ", text)).strip()


def mask_texts(texts, nlp, batch_size=2000):
    """Returns (masked_normalised_texts, per-type replacement counts)."""
    pre = [pre_mask(t) for t in texts]
    counts = {"HASHTAG": sum(t.count("HASHTAG") for t in pre)}
    out = []
    for doc in nlp.pipe(pre, batch_size=batch_size):
        text, last = [], 0
        for ent in doc.ents:
            if ent.label_ not in ENTITY_MAP or ent.text.strip("@#") in PLACEHOLDERS or set(ent.text.split()) & PLACEHOLDERS:
                continue
            tag = ENTITY_MAP[ent.label_]
            counts[tag] = counts.get(tag, 0) + 1
            text.append(doc.text[last:ent.start_char]); text.append(tag); last = ent.end_char
        text.append(doc.text[last:])
        out.append(normalize_for_bertweet("".join(text)))
    return out, counts


def load_bertweet(device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("vinai/bertweet-base", revision=REVISION, normalization=False)
    model = AutoModel.from_pretrained("vinai/bertweet-base", revision=REVISION).to(device).eval()
    return tok, model


@torch.inference_mode()
def embed(texts, tok, model, device, batch_size=64, max_length=128):
    out = np.zeros((len(texts), 768), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
    for s in range(0, len(order), batch_size):
        rows = order[s:s + batch_size]
        enc = tok([texts[i] for i in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[rows] = ((h * m).sum(1) / m.sum(1).clamp(min=1.0)).float().cpu().numpy()
    return out


def post_texts():
    """Raw text in ids.csv row order."""
    posts = pd.read_csv("data/processed/posts.csv", dtype=str, keep_default_na=False).set_index("post_id")
    with open(FULL / "ids.csv", newline="", encoding="utf-8") as f:
        ids = [r["post_id"] for r in sorted(csv.DictReader(f), key=lambda r: int(r["embedding_index"]))]
    return ids, posts.loc[ids, "text"].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/nlp/embeddings/masked")
    parser.add_argument("--check", type=int, default=0, help="only embed N random rows unmasked and compare to full/")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids, raw = post_texts()
    tok, model = load_bertweet(device)
    if args.check:
        rows = np.random.default_rng(0).choice(len(raw), args.check, replace=False)
        new = embed([normalize_for_bertweet(raw[i]) for i in rows], tok, model, device)
        old = np.load(FULL / "embeddings.npy", mmap_mode="r")[np.sort(rows)][np.argsort(np.argsort(rows))]
        cos = (new * old).sum(1) / np.linalg.norm(new, axis=1) / np.linalg.norm(old, axis=1)
        print(f"reproduction check on {args.check} rows: cosine min={cos.min():.5f} mean={cos.mean():.6f}")
        return
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    t0 = time.time()
    masked, counts = mask_texts(raw, nlp)
    print(f"masked {len(raw)} posts in {time.time() - t0:.0f}s; replacements {counts}", flush=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"post_id": ids, "text_raw": raw, "text_masked": masked}).to_csv(out / "masked_texts.csv", index=False)
    t0 = time.time()
    emb = embed(masked, tok, model, device)
    np.save(out / "embeddings.npy", emb)
    shutil.copy(FULL / "ids.csv", out / "ids.csv")
    (out / "metadata.json").write_text(json.dumps({
        "source": "experiments/generalization/masking.py", "model_revision": REVISION, "max_length": 128,
        "pooling": "attention-mask mean of last hidden layer", "entity_map": ENTITY_MAP, "replacements": counts,
        "spacy_model": "en_core_web_lg " + nlp.meta["version"], "embed_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
