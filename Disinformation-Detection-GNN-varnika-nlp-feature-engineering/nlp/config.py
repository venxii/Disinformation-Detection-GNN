from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Inputs produced by Person 1 (read-only for the NLP component)
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
POSTS_FILE = PROCESSED_DATA / "posts.csv"
CLAIMS_FILE = PROCESSED_DATA / "claims.csv"

GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
GRAPH_METADATA_FILE = GRAPHS_DIR / "graph_metadata.csv"

# Outputs produced by the NLP component
NLP_DATA = PROJECT_ROOT / "data" / "nlp"
POSTS_CLEAN_FILE = NLP_DATA / "posts_clean.csv"
POSTS_CLEAN_INFO_FILE = NLP_DATA / "posts_clean_info.json"
TFIDF_DIR = NLP_DATA / "tfidf"

EMBEDDINGS_DIR = NLP_DATA / "embeddings"
PILOT_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "pilot"
FULL_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "full"
FULL_CHECKPOINT_DIR = NLP_DATA / "checkpoints" / "bertweet_full"
TOKENIZER_CHECK_FILE = NLP_DATA / "bertweet_tokenizer_check.json"

# Downloaded model files stay inside the project (git-ignored)
HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"


# ============================================================
# PREPROCESSING
# ============================================================

# Bump this whenever a cleaning rule changes, so downstream
# features can record which text version they were built from.
PREPROCESSING_VERSION = "1.1"


# ============================================================
# BERTWEET
# ============================================================

BERTWEET_MODEL_NAME = "vinai/bertweet-base"

# Commit of the model repository on the Hugging Face Hub ("main" when
# first downloaded). Pinning it keeps embeddings reproducible.
BERTWEET_REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"

# BERTweet has 130 position embeddings: 128 tokens + 2 reserved offsets
BERTWEET_MAX_LENGTH = 128
EMBEDDING_DIM = 768
