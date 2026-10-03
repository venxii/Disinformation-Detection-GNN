"""
Frozen BERTweet sentence embeddings for PHEME posts.

Each post's `text_bertweet` (already normalised in Stage 2) is fed to
the pre-trained vinai/bertweet-base model. The model is never trained:
it is put in evaluation mode, all weights are frozen and every forward
pass runs under torch.inference_mode().

One 768-dimensional vector per post is produced by attention-mask-aware
mean pooling over the last hidden layer (see mean_pool).

From the project root:
    .venv/bin/python -m nlp.bertweet_embeddings pilot   # 100-post sample
    .venv/bin/python -m nlp.bertweet_embeddings full    # every post, resumable

Loading the full output:
    from nlp.bertweet_embeddings import load_full_embeddings
    embeddings, ids, metadata = load_full_embeddings()
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from nlp.config import (
    BERTWEET_MAX_LENGTH,
    BERTWEET_MODEL_NAME,
    BERTWEET_REVISION,
    EMBEDDING_DIM,
    FULL_CHECKPOINT_DIR,
    FULL_EMBEDDINGS_DIR,
    HF_CACHE_DIR,
    PILOT_EMBEDDINGS_DIR,
    POSTS_CLEAN_FILE,
    POSTS_CLEAN_INFO_FILE,
    PREPROCESSING_VERSION,
)


TEXT_COLUMN = "text_bertweet"

DEFAULT_BATCH_SIZE = 32
PILOT_SAMPLE_SIZE = 100
PILOT_SEED = 42

POOLING_DESCRIPTION = (
    "attention-mask-aware mean of the last hidden layer over all "
    "non-padding tokens (including <s> and </s>)"
)


# ============================================================
# DEVICE AND MODEL
# ============================================================

def select_device(preferred="auto"):
    """'auto' -> 'mps' when available, otherwise 'cpu'."""

    if preferred == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"

    if preferred == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    return preferred


def freeze_model(model):
    """Evaluation mode + no gradients: weights can never be updated."""

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def load_bertweet(device, revision=BERTWEET_REVISION, cache_dir=HF_CACHE_DIR):
    """
    Load the BERTweet tokenizer and frozen model.

    normalization=False because text_bertweet is already normalised;
    the tokenizer then only splits on spaces and applies BPE.
    """

    def from_cache_or_hub(loader, **kwargs):
        try:
            return loader(local_files_only=True, **kwargs)
        except OSError:
            return loader(**kwargs)

    common = {
        "pretrained_model_name_or_path": BERTWEET_MODEL_NAME,
        "revision": revision,
        "cache_dir": cache_dir,
    }

    tokenizer = from_cache_or_hub(
        AutoTokenizer.from_pretrained, normalization=False, **common
    )
    model = from_cache_or_hub(AutoModel.from_pretrained, **common)

    freeze_model(model)
    model.to(device)

    return tokenizer, model


def model_device(model):
    """The device the model weights actually live on."""

    return str(next(model.parameters()).device)


# ============================================================
# POOLING AND EMBEDDING
# ============================================================

def mean_pool(last_hidden_state, attention_mask):
    """
    Average the token vectors of each text, ignoring padding.

    last_hidden_state: (batch, tokens, 768)
    attention_mask:    (batch, tokens), 1 = real token, 0 = padding

    Padding vectors are multiplied by 0 and the sum is divided by the
    number of real tokens. A row with no real tokens returns zeros
    instead of dividing by zero.
    """

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)

    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)

    return summed / counts


def prepare_texts(texts):
    """Missing or whitespace-only text becomes ''."""

    prepared = []

    for text in texts:

        if text is None or (isinstance(text, float) and np.isnan(text)):
            prepared.append("")
        else:
            prepared.append(str(text).strip())

    return prepared


def embed_texts(
    texts,
    tokenizer,
    model,
    device,
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=BERTWEET_MAX_LENGTH,
    sort_by_length=True,
):
    """
    Return a float32 array with one row per input text, in input order.

    sort_by_length=True groups texts of similar length into batches to
    reduce padding; sort_by_length=False batches texts in input order.
    Either way every pooled vector is written back to its original row.
    An empty text is encoded as just <s></s>, which gives a valid vector.
    """

    texts = prepare_texts(texts)
    embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    if sort_by_length:
        order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
    else:
        order = list(range(len(texts)))

    with torch.inference_mode():

        for start in range(0, len(order), batch_size):

            rows = order[start:start + batch_size]

            encoded = tokenizer(
                [texts[i] for i in rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool(output.last_hidden_state, attention_mask)

            embeddings[rows] = pooled.float().cpu().numpy()

    return embeddings


def token_lengths(texts, tokenizer):
    """Number of tokens per text (including <s> and </s>), without truncation."""

    return [
        len(tokenizer(text, truncation=False)["input_ids"])
        for text in prepare_texts(texts)
    ]


# ============================================================
# SELECTING POSTS
# ============================================================

def load_clean_posts(path=POSTS_CLEAN_FILE):
    """Load posts_clean.csv with every column as an exact string."""

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv/bin/python -m nlp.build_clean_text"
        )

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def select_pilot_posts(posts, sample_size=PILOT_SAMPLE_SIZE, seed=PILOT_SEED):
    """
    Deterministic random sample (fixed seed), returned in posts_clean.csv
    order. If there are fewer posts than sample_size, all are returned.
    """

    if len(posts) <= sample_size:
        selected = posts
    else:
        selected = posts.sample(n=sample_size, random_state=seed).sort_index()

    return selected.reset_index(drop=True)


# ============================================================
# SAVING AND LOADING
# ============================================================

def id_mapping(subset, index_column="row_index"):
    """<index_column> -> post_id / root_id for an embedding matrix."""

    return pd.DataFrame({
        index_column: range(len(subset)),
        "post_id": subset["post_id"].tolist(),
        "root_id": subset["root_id"].tolist(),
    })


def save_embeddings(output_dir, embeddings, subset, metadata,
                    index_column="row_index"):
    """Write embeddings.npy, ids.csv and metadata.json."""

    if embeddings.shape != (len(subset), EMBEDDING_DIM):
        raise ValueError(
            f"embeddings shape {embeddings.shape} does not match "
            f"({len(subset)}, {EMBEDDING_DIM})"
        )

    if subset["post_id"].duplicated().any():
        raise ValueError("subset contains duplicate post_id values")

    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_save_npy(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    atomic_write_text(
        output_dir / "ids.csv",
        id_mapping(subset, index_column).to_csv(index=False),
    )
    atomic_write_text(output_dir / "metadata.json", json.dumps(metadata, indent=2))


def load_embeddings(output_dir, index_column="row_index", mmap=False):
    """Load embeddings and their ID mapping, checking that they line up."""

    embeddings = np.load(
        output_dir / "embeddings.npy", mmap_mode="r" if mmap else None
    )
    ids = pd.read_csv(output_dir / "ids.csv", dtype=str, keep_default_na=False)

    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"embeddings have {embeddings.shape[0]} rows but "
            f"ids.csv has {len(ids)}"
        )

    if ids[index_column].tolist() != [str(i) for i in range(len(ids))]:
        raise ValueError(f"ids.csv {index_column} column is not 0..n-1")

    if ids["post_id"].duplicated().any():
        raise ValueError("ids.csv contains duplicate post_id values")

    return embeddings, ids


def load_full_embeddings(output_dir=FULL_EMBEDDINGS_DIR, mmap=False):
    """
    Load the full-dataset embeddings. Refuses outputs whose metadata does
    not record a completed, validated extraction.
    """

    metadata_file = output_dir / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found: the full extraction has not completed"
        )

    with open(metadata_file, encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("status") != "complete":
        raise RuntimeError(
            f"Full extraction status is '{metadata.get('status')}', not 'complete'"
        )

    embeddings, ids = load_embeddings(
        output_dir, index_column="embedding_index", mmap=mmap
    )

    if embeddings.shape != (metadata["num_posts"], metadata["embedding_dim"]):
        raise ValueError(
            f"embeddings shape {embeddings.shape} does not match metadata"
        )

    return embeddings, ids, metadata


# ============================================================
# SAFE (ATOMIC) WRITES
# ============================================================
# Each file is written under a temporary name, flushed to disk and then
# renamed. A rename is atomic, so a crash leaves either the complete old
# file, the complete new file, or a leftover *.tmp file - never a
# half-written file under the real name.

def _atomic_write(path, write):

    tmp_path = path.with_name(path.name + ".tmp")

    with open(tmp_path, "wb") as file:
        write(file)
        file.flush()
        os.fsync(file.fileno())

    os.replace(tmp_path, path)


def atomic_write_text(path, text):
    _atomic_write(path, lambda file: file.write(text.encode("utf-8")))


def atomic_save_npy(path, array):
    _atomic_write(path, lambda file: np.save(file, array))


def atomic_save_npz(path, **arrays):
    _atomic_write(path, lambda file: np.savez(file, **arrays))


# ============================================================
# CHECKPOINTED EXTRACTION
# ============================================================

FULL_CHUNK_SIZE = 1024
RUN_CONFIG_FILE = "run_config.json"
PROGRESS_FILE = "progress.json"


class CheckpointMismatchError(RuntimeError):
    """Existing checkpoints were made with different inputs or settings."""


def file_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def ids_sha256(post_ids):
    return hashlib.sha256("\n".join(post_ids).encode("utf-8")).hexdigest()


def extraction_config(post_ids, input_sha256, batch_size, chunk_size, device):
    """Everything that must be identical for checkpoints to be reusable."""

    return {
        "model_name": BERTWEET_MODEL_NAME,
        "model_revision": BERTWEET_REVISION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "text_column": TEXT_COLUMN,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float32",
        "pooling": POOLING_DESCRIPTION,
        "max_length": BERTWEET_MAX_LENGTH,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "sort_by_length": False,
        "device": device,
        "num_posts": len(post_ids),
        "input_sha256": input_sha256,
        "post_ids_sha256": ids_sha256(post_ids),
    }


def chunk_bounds(num_posts, chunk_size):
    """[(start, end), ...] covering 0..num_posts in order, no gaps or overlaps."""

    return [
        (start, min(start + chunk_size, num_posts))
        for start in range(0, num_posts, chunk_size)
    ]


def chunk_file(checkpoint_dir, index):
    return checkpoint_dir / f"chunk_{index:05d}.npz"


def load_valid_chunk(path, expected_ids):
    """
    Return (embeddings, None) for a complete, correct chunk, otherwise
    (None, reason). A chunk is only trusted if it loads, has the right
    shape and dtype, contains finite values and exactly the expected IDs.
    """

    try:
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            ids = data["post_ids"].tolist()
    except Exception as error:
        return None, f"unreadable ({type(error).__name__}: {error})"

    if embeddings.shape != (len(expected_ids), EMBEDDING_DIM):
        return None, f"wrong shape {embeddings.shape}"

    if embeddings.dtype != np.float32:
        return None, f"wrong dtype {embeddings.dtype}"

    if ids != list(expected_ids):
        return None, "post_ids do not match the expected rows"

    if not np.isfinite(embeddings).all():
        return None, "contains NaN or Inf"

    return embeddings, None


def prepare_checkpoint_dir(checkpoint_dir, config, log=print):
    """
    Create the checkpoint folder or verify that existing checkpoints were
    made with an identical configuration. Leftover *.tmp files (from an
    interrupted write) are removed; they were never marked as complete.
    """

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / RUN_CONFIG_FILE

    if config_path.exists():

        with open(config_path, encoding="utf-8") as file:
            saved = json.load(file)

        differences = {
            key: {"checkpoint": saved.get(key), "current": config.get(key)}
            for key in sorted(set(saved) | set(config))
            if saved.get(key) != config.get(key)
        }

        if differences:
            raise CheckpointMismatchError(
                "Existing checkpoints were created with different settings:\n"
                + json.dumps(differences, indent=2)
            )

        log(f"Resuming from checkpoints in {checkpoint_dir}")

    else:
        atomic_write_text(config_path, json.dumps(config, indent=2))

    for tmp_path in sorted(checkpoint_dir.glob("*.tmp")):
        log(f"Removing incomplete temporary file {tmp_path.name}")
        tmp_path.unlink()


def extract_with_checkpoints(
    post_ids, texts, embed_fn, checkpoint_dir, config,
    chunk_size=FULL_CHUNK_SIZE, log=print,
):
    """
    Embed `texts` chunk by chunk in input order, saving each finished chunk
    atomically. Valid existing chunks are reused; invalid ones are reported,
    deleted and recomputed. Returns (embeddings, stats).

    embed_fn(list_of_texts) -> float32 array (len, EMBEDDING_DIM)
    """

    if len(post_ids) != len(texts):
        raise ValueError("post_ids and texts have different lengths")

    prepare_checkpoint_dir(checkpoint_dir, config, log)

    bounds = chunk_bounds(len(post_ids), chunk_size)

    stats = {
        "chunks_total": len(bounds),
        "chunks_reused": 0,
        "chunks_computed": 0,
        "chunks_recomputed_invalid": 0,
        "posts_computed": 0,
    }

    completed = []
    start_time = datetime.now(timezone.utc)

    for index, (start, end) in enumerate(bounds):

        path = chunk_file(checkpoint_dir, index)
        expected_ids = post_ids[start:end]

        if path.exists():

            _, problem = load_valid_chunk(path, expected_ids)

            if problem is None:
                stats["chunks_reused"] += 1
                completed.append(index)
                continue

            log(f"Chunk {index} is invalid ({problem}); deleting and recomputing it.")
            path.unlink()
            stats["chunks_recomputed_invalid"] += 1

        embeddings = np.asarray(embed_fn(texts[start:end]), dtype=np.float32)

        if embeddings.shape != (end - start, EMBEDDING_DIM):
            raise ValueError(
                f"chunk {index}: embed_fn returned shape {embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(f"chunk {index}: embeddings contain NaN or Inf")

        atomic_save_npz(
            path, embeddings=embeddings, post_ids=np.array(expected_ids)
        )

        completed.append(index)
        stats["chunks_computed"] += 1
        stats["posts_computed"] += end - start

        atomic_write_text(
            checkpoint_dir / PROGRESS_FILE,
            json.dumps({
                "completed_chunks": len(completed),
                "chunks_total": len(bounds),
                "last_completed_chunk": index,
                "last_completed_row": end - 1,
                "last_completed_post_id": expected_ids[-1],
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2),
        )

        if stats["chunks_computed"] % 10 == 0 or index == len(bounds) - 1:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            rate = stats["posts_computed"] / max(elapsed, 1e-9)
            remaining = (len(post_ids) - end) / max(rate, 1e-9)
            log(f"  chunk {index + 1}/{len(bounds)}  rows {end}/{len(post_ids)}  "
                f"{rate:.0f} posts/s  ~{remaining / 60:.1f} min left")

    embeddings, assembled_ids = assemble_chunks(checkpoint_dir, post_ids, chunk_size)

    stats["assembled_ids_match_input"] = assembled_ids == list(post_ids)
    stats["chunk_indices"] = sorted(completed) == list(range(len(bounds)))

    return embeddings, stats


def assemble_chunks(checkpoint_dir, post_ids, chunk_size):
    """Concatenate all chunks in order, re-validating each one."""

    bounds = chunk_bounds(len(post_ids), chunk_size)
    chunk_files = sorted(checkpoint_dir.glob("chunk_*.npz"))

    if len(chunk_files) != len(bounds):
        raise RuntimeError(
            f"expected {len(bounds)} chunk files, found {len(chunk_files)}"
        )

    embeddings = np.empty((len(post_ids), EMBEDDING_DIM), dtype=np.float32)
    assembled_ids = []

    for index, (start, end) in enumerate(bounds):

        chunk, problem = load_valid_chunk(
            chunk_file(checkpoint_dir, index), post_ids[start:end]
        )

        if problem is not None:
            raise RuntimeError(f"chunk {index} failed re-validation: {problem}")

        embeddings[start:end] = chunk
        assembled_ids.extend(post_ids[start:end])

    return embeddings, assembled_ids


def completed_output_exists(output_dir):
    """True if output_dir holds an extraction marked as complete."""

    metadata_file = output_dir / "metadata.json"

    if not metadata_file.exists():
        return False

    try:
        with open(metadata_file, encoding="utf-8") as file:
            return json.load(file).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


# ============================================================
# PILOT CHECKS
# ============================================================

def parameter_fingerprint(model):
    """
    SHA-256 of the raw bytes of every weight; identical before/after
    means no weight changed. Weights are copied to the CPU without any
    dtype conversion (a combined MPS->CPU + float64 copy gave wrong values).
    """

    digest = hashlib.sha256()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().to("cpu").contiguous().numpy().tobytes())

    return digest.hexdigest()


def run_pilot_checks(
    posts, selected, embeddings, tokenizer, model, device,
    sample_size, seed, fingerprint_before,
):
    """Return (checks: name -> bool, details: name -> value)."""

    details = {}

    reselected = select_pilot_posts(posts, sample_size, seed)

    # Padding check: embedding a text alone (no padding) must match its
    # embedding inside a padded batch.
    probe_texts = selected[TEXT_COLUMN].tolist()[:8]
    alone = np.vstack([
        embed_texts([text], tokenizer, model, device) for text in probe_texts
    ])
    padding_diff = float(np.abs(alone - embeddings[:len(probe_texts)]).max())
    details["padding_max_abs_diff"] = padding_diff

    repeat = embed_texts(selected[TEXT_COLUMN].tolist(), tokenizer, model, device)
    repeat_diff = float(np.abs(repeat - embeddings).max())
    details["repeat_run_max_abs_diff"] = repeat_diff

    empty = embed_texts(["", "   "], tokenizer, model, device)

    fingerprint_after = parameter_fingerprint(model)

    # The same texts on the CPU must give (almost) the same vectors
    if device != "cpu":
        model.to("cpu")
        on_cpu = embed_texts(probe_texts, tokenizer, model, "cpu")
        model.to(device)
        details["cpu_vs_device_max_abs_diff"] = float(
            np.abs(on_cpu - embeddings[:len(probe_texts)]).max()
        )
    else:
        details["cpu_vs_device_max_abs_diff"] = 0.0

    details["actual_device"] = model_device(model)
    details["mps_fallback_env"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")

    expected_count = min(sample_size, len(posts))
    selected_ids = selected["post_id"].tolist()

    checks = {
        "selected count": len(selected) == expected_count,
        "embedding shape": embeddings.shape == (expected_count, EMBEDDING_DIM),
        "no NaN or Inf": bool(np.isfinite(embeddings).all()),
        "no duplicate post_id": len(set(selected_ids)) == len(selected_ids),
        "no empty post_id": all(selected_ids),
        "post_ids exist in posts_clean": set(selected_ids) <= set(posts["post_id"]),
        "same seed -> same ids": reselected["post_id"].tolist() == selected_ids,
        "padding excluded (diff < 1e-3)": padding_diff < 1e-3,
        "repeat run matches (diff < 1e-3)": repeat_diff < 1e-3,
        "empty text gives finite vectors": (
            empty.shape == (2, EMBEDDING_DIM) and bool(np.isfinite(empty).all())
        ),
        "model in eval mode": not model.training,
        "all parameters frozen": not any(
            p.requires_grad for p in model.parameters()
        ),
        "weights unchanged (SHA-256)": fingerprint_after == fingerprint_before,
        "device matches CPU (diff < 1e-3)": (
            details["cpu_vs_device_max_abs_diff"] < 1e-3
        ),
        "runs on requested device": details["actual_device"].startswith(device),
    }

    return checks, details


# ============================================================
# FULL-RUN CHECKS
# ============================================================

def run_full_checks(
    posts, output_dir, stats, tokenizer, model, device,
    fingerprint_before, expected_posts, spot_check_size=64, seed=0,
):
    """
    Validate the written embeddings.npy and ids.csv against posts_clean.csv.
    Returns (checks: name -> bool, details: name -> value).
    """

    details = {}

    embeddings, ids = load_embeddings(
        output_dir, index_column="embedding_index", mmap=True
    )

    num_posts = len(posts)
    post_ids = ids["post_id"].tolist()

    # Re-embed random rows and compare with the stored rows: proves that
    # row i of the matrix really belongs to post_id i of ids.csv.
    rng = np.random.default_rng(seed)
    spot_rows = sorted(rng.choice(num_posts, size=spot_check_size, replace=False))
    spot_texts = posts[TEXT_COLUMN].iloc[spot_rows].tolist()
    recomputed = embed_texts(spot_texts, tokenizer, model, device, sort_by_length=False)
    details["spot_check_rows"] = len(spot_rows)
    details["spot_check_max_abs_diff"] = float(
        np.abs(recomputed - embeddings[spot_rows]).max()
    )

    # Plain average over all tokens of a single, unpadded text must equal
    # the stored row: confirms attention-mask-aware mean pooling.
    pooling_diffs = []

    with torch.inference_mode():
        for row in spot_rows[:5]:
            encoded = tokenizer(
                posts[TEXT_COLUMN].iloc[row], truncation=True,
                max_length=BERTWEET_MAX_LENGTH, return_tensors="pt",
            )
            hidden = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            ).last_hidden_state
            manual = hidden.mean(dim=1).float().cpu().numpy()[0]
            pooling_diffs.append(float(np.abs(manual - embeddings[row]).max()))

    details["manual_mean_pool_max_abs_diff"] = max(pooling_diffs)

    # Pilot rows must match the same posts in the full matrix
    pilot_diff = None

    if (PILOT_EMBEDDINGS_DIR / "embeddings.npy").exists():
        pilot_embeddings, pilot_ids = load_embeddings(PILOT_EMBEDDINGS_DIR)
        row_of = {post_id: row for row, post_id in enumerate(post_ids)}
        rows = [row_of[post_id] for post_id in pilot_ids["post_id"]]
        pilot_diff = float(np.abs(pilot_embeddings - embeddings[rows]).max())

    details["pilot_vs_full_max_abs_diff"] = pilot_diff

    finite = all(
        np.isfinite(embeddings[start:start + 8192]).all()
        for start in range(0, num_posts, 8192)
    )

    checks = {
        "1. shape is (expected posts, 768)": (
            embeddings.shape == (expected_posts, EMBEDDING_DIM)
            and num_posts == expected_posts
        ),
        "2. dtype is float32": embeddings.dtype == np.float32,
        "3. no NaN or Inf": bool(finite),
        "4. ids.csv has one row per post": len(ids) == expected_posts,
        "5. no missing or duplicate post_id": (
            all(post_ids) and len(set(post_ids)) == len(post_ids)
        ),
        "6. post_id and root_id match posts_clean row by row": (
            post_ids == posts["post_id"].tolist()
            and ids["root_id"].tolist() == posts["root_id"].tolist()
        ),
        "7. matrix rows match ID mapping (re-embedded spot check)": (
            details["spot_check_max_abs_diff"] < 1e-3
            and (pilot_diff is None or pilot_diff < 1e-3)
        ),
        "8. every post processed exactly once": bool(
            stats["assembled_ids_match_input"]
        ),
        "9. no chunks skipped or duplicated": bool(
            stats["chunk_indices"]
            and stats["chunks_reused"] + stats["chunks_computed"]
            == stats["chunks_total"]
        ),
        "11. model weights frozen and unchanged": (
            not any(p.requires_grad for p in model.parameters())
            and parameter_fingerprint(model) == fingerprint_before
        ),
        "12. model in evaluation mode": not model.training,
        "13. attention-mask-aware mean pooling (manual check)": (
            details["manual_mean_pool_max_abs_diff"] < 1e-3
        ),
    }

    return checks, details


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args(argv=None):

    parser = argparse.ArgumentParser(description="BERTweet embeddings")
    commands = parser.add_subparsers(dest="command", required=True)

    pilot = commands.add_parser("pilot", help="Embed a small random sample")
    pilot.add_argument("--sample-size", type=int, default=PILOT_SAMPLE_SIZE)
    pilot.add_argument("--seed", type=int, default=PILOT_SEED)
    pilot.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    pilot.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    pilot.add_argument("--overwrite", action="store_true")

    full = commands.add_parser("full", help="Embed every post (checkpointed)")
    full.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    full.add_argument("--chunk-size", type=int, default=FULL_CHUNK_SIZE)
    full.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    full.add_argument("--overwrite", action="store_true",
                      help="Replace an existing completed extraction")
    full.add_argument("--restart", action="store_true",
                      help="Discard existing checkpoints and start from scratch")
    full.add_argument("--keep-checkpoints", action="store_true",
                      help="Keep chunk files after a successful run")

    return parser.parse_args(argv)


def run_pilot(args):

    output_dir = PILOT_EMBEDDINGS_DIR

    print("=" * 60)
    print("BERTWEET EMBEDDING PILOT")
    print("=" * 60)

    if (output_dir / "embeddings.npy").exists() and not args.overwrite:
        print(f"\nERROR: {output_dir} already has a pilot. Use --overwrite.")
        sys.exit(1)

    device = select_device(args.device)
    print(f"\nRequested device : {device}")

    posts = load_clean_posts()
    selected = select_pilot_posts(posts, args.sample_size, args.seed)
    print(f"Posts selected   : {len(selected)} (seed {args.seed})")

    tokenizer, model = load_bertweet(device)
    print(f"Model device     : {model_device(model)}")

    fingerprint_before = parameter_fingerprint(model)
    texts = selected[TEXT_COLUMN].tolist()

    try:
        embeddings = embed_texts(
            texts, tokenizer, model, device, batch_size=args.batch_size
        )
    except (RuntimeError, NotImplementedError) as error:
        print(f"\nERROR during forward pass on {device}: {error}")
        print("Re-run with --device cpu if this is an unsupported MPS operation.")
        sys.exit(1)

    lengths = token_lengths(texts, tokenizer)
    truncated = sum(length > BERTWEET_MAX_LENGTH for length in lengths)
    unk_tokens = sum(
        (np.array(tokenizer(t)["input_ids"]) == tokenizer.unk_token_id).sum()
        for t in prepare_texts(texts)
    )

    checks, details = run_pilot_checks(
        posts, selected, embeddings, tokenizer, model, device,
        args.sample_size, args.seed, fingerprint_before,
    )

    print("\nValidation:")
    for name, passed in checks.items():
        print(f"  {name:<34}: {'OK' if passed else 'FAILED'}")

    print("\nDetails:")
    print(f"  actual device                    : {details['actual_device']}")
    print(f"  padding max abs diff             : {details['padding_max_abs_diff']:.2e}")
    print(f"  repeat run max abs diff          : {details['repeat_run_max_abs_diff']:.2e}")
    print(f"  CPU vs {device} max abs diff        : {details['cpu_vs_device_max_abs_diff']:.2e}")
    print(f"  tokens per post (min/mean/max)   : "
          f"{min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}")
    print(f"  posts truncated at {BERTWEET_MAX_LENGTH} tokens     : {truncated}")
    print(f"  <unk> tokens in sample           : {int(unk_tokens)}")
    print(f"  embedding norm (min/mean/max)    : "
          f"{np.linalg.norm(embeddings, axis=1).min():.2f}/"
          f"{np.linalg.norm(embeddings, axis=1).mean():.2f}/"
          f"{np.linalg.norm(embeddings, axis=1).max():.2f}")

    metadata = {
        "stage": "pilot",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": BERTWEET_MODEL_NAME,
        "model_revision": BERTWEET_REVISION,
        "weights_format": "safetensors (identical to pytorch_model.bin)",
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float32",
        "pooling": POOLING_DESCRIPTION,
        "layer": "last hidden layer",
        "max_length": BERTWEET_MAX_LENGTH,
        "truncation": True,
        "batch_size": args.batch_size,
        "tokenizer_normalization": False,
        "text_column": TEXT_COLUMN,
        "preprocessing_version": PREPROCESSING_VERSION,
        "sample_size": len(selected),
        "seed": args.seed,
        "selection": "pandas DataFrame.sample(random_state=seed), "
                     "rows kept in posts_clean.csv order",
        "requested_device": device,
        "actual_device": details["actual_device"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "token_length_min_mean_max": [
            int(min(lengths)), float(np.mean(lengths)), int(max(lengths))
        ],
        "truncated_posts": int(truncated),
        "unk_tokens": int(unk_tokens),
        "checks": checks,
        "check_details": details,
    }

    save_embeddings(output_dir, embeddings, selected, metadata)
    load_embeddings(output_dir)

    print(f"\nSaved to: {output_dir}")
    print("\n" + "=" * 60)

    if all(checks.values()):
        print("PILOT PASSED")
    else:
        print("PILOT FAILED VALIDATION")
        sys.exit(1)

    print("=" * 60)


def run_full(args):

    output_dir = FULL_EMBEDDINGS_DIR
    checkpoint_dir = FULL_CHECKPOINT_DIR

    print("=" * 60)
    print("BERTWEET FULL EXTRACTION")
    print("=" * 60)

    if completed_output_exists(output_dir) and not args.overwrite:
        print(f"\nERROR: {output_dir} already holds a completed extraction.")
        print("Use --overwrite to replace it.")
        sys.exit(1)

    if args.restart and checkpoint_dir.exists():
        print(f"\n--restart: removing existing checkpoints in {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    device = select_device(args.device)

    posts = load_clean_posts()
    post_ids = posts["post_id"].tolist()
    texts = posts[TEXT_COLUMN].tolist()

    with open(POSTS_CLEAN_INFO_FILE, encoding="utf-8") as file:
        expected_posts = json.load(file)["summary"]["posts"]

    input_sha256 = file_sha256(POSTS_CLEAN_FILE)
    config = extraction_config(
        post_ids, input_sha256, args.batch_size, args.chunk_size, device
    )

    print(f"\nPosts            : {len(posts)} (expected {expected_posts})")
    print(f"Requested device : {device}")

    tokenizer, model = load_bertweet(device)
    actual_device = model_device(model)
    print(f"Model device     : {actual_device}")
    print(f"MPS CPU fallback : {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') or 'disabled'}")

    fingerprint_before = parameter_fingerprint(model)

    def embed_fn(chunk_texts):
        return embed_texts(
            chunk_texts, tokenizer, model, device,
            batch_size=args.batch_size, sort_by_length=False,
        )

    started = datetime.now(timezone.utc)
    print(f"\nExtraction started {started.isoformat(timespec='seconds')}")

    try:
        embeddings, stats = extract_with_checkpoints(
            post_ids, texts, embed_fn, checkpoint_dir, config,
            chunk_size=args.chunk_size,
        )
    except CheckpointMismatchError as error:
        print(f"\nERROR: {error}")
        print("Nothing was reused. Run with --restart to discard these checkpoints.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed chunks are saved; re-run the same "
              "command to resume.")
        sys.exit(130)
    except (RuntimeError, NotImplementedError) as error:
        print(f"\nERROR during extraction on {device}: {error}")
        print("Completed chunks are kept. No automatic CPU fallback was used.")
        print("To retry on CPU explicitly: --device cpu --restart")
        sys.exit(1)

    duration = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\nExtraction finished in {duration / 60:.1f} min "
          f"({stats['chunks_computed']} chunks computed, "
          f"{stats['chunks_reused']} reused, "
          f"{stats['chunks_recomputed_invalid']} recomputed after validation failure)")

    print("\nCounting tokens for truncation statistics...")
    lengths = np.array(token_lengths(texts, tokenizer))
    truncated_ids = [
        post_ids[i] for i in np.flatnonzero(lengths > BERTWEET_MAX_LENGTH)
    ]

    # Remove a previous completion marker first, so a crash while writing
    # can never leave old "complete" metadata next to new files.
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "metadata.json"

    if metadata_file.exists():
        metadata_file.unlink()

    atomic_save_npy(output_dir / "embeddings.npy", embeddings)
    atomic_write_text(
        output_dir / "ids.csv",
        id_mapping(posts, "embedding_index").to_csv(index=False),
    )

    print("Validating written files...")
    checks, details = run_full_checks(
        posts, output_dir, stats, tokenizer, model, device,
        fingerprint_before, expected_posts,
    )

    metadata = {
        "stage": "full",
        "status": "complete" if all(checks.values()) else "failed_validation",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extraction_started_at": started.isoformat(timespec="seconds"),
        "extraction_duration_seconds": round(duration, 1),
        "model_name": BERTWEET_MODEL_NAME,
        "model_revision": BERTWEET_REVISION,
        "weights_format": "safetensors (identical to pytorch_model.bin)",
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float32",
        "pooling": POOLING_DESCRIPTION,
        "layer": "last hidden layer",
        "max_length": BERTWEET_MAX_LENGTH,
        "truncation": True,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "batching": "input order (no length sorting)",
        "tokenizer_normalization": False,
        "text_column": TEXT_COLUMN,
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_file": "data/nlp/posts_clean.csv",
        "input_sha256": input_sha256,
        "post_ids_sha256": config["post_ids_sha256"],
        "num_posts": len(posts),
        "completed_posts": int(embeddings.shape[0]),
        "requested_device": device,
        "actual_device": actual_device,
        "mps_cpu_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "id_file_columns": ["embedding_index", "post_id", "root_id"],
        "checkpoint_stats": stats,
        "warnings": {
            "truncated_posts": len(truncated_ids),
            "truncated_post_ids": truncated_ids,
            "token_length_max": int(lengths.max()),
            "empty_text_posts": int(sum(1 for t in texts if not t.strip())),
        },
        "checks": checks,
        "check_details": details,
        "checkpoint_recovery": (
            "covered by unit tests (interrupt/resume, corrupted chunk, "
            "settings mismatch, leftover temp file); "
            f"this run reused {stats['chunks_reused']} chunks"
        ),
    }

    atomic_write_text(metadata_file, json.dumps(metadata, indent=2))

    print("\nValidation:")
    for name, passed in checks.items():
        print(f"  {name:<58}: {'OK' if passed else 'FAILED'}")

    print("\nDetails:")
    for name, value in details.items():
        print(f"  {name:<32}: {value}")
    print(f"  truncated posts                 : {len(truncated_ids)}")

    if not all(checks.values()):
        print("\nFULL EXTRACTION FAILED VALIDATION (status: failed_validation).")
        print(f"Checkpoints kept in {checkpoint_dir}")
        sys.exit(1)

    _, _, reloaded = load_full_embeddings(output_dir, mmap=True)
    print(f"\n  14. metadata records completion : {reloaded['status'] == 'complete'}")

    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir)
        print(f"Checkpoints removed: {checkpoint_dir}")

    print(f"\nSaved to: {output_dir}")
    print("\n" + "=" * 60)
    print("FULL EXTRACTION COMPLETE")
    print("=" * 60)


def main(argv=None):

    args = parse_args(argv)

    if args.command == "pilot":
        run_pilot(args)
    elif args.command == "full":
        run_full(args)


if __name__ == "__main__":
    main()
