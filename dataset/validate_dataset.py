import pandas as pd
import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

POSTS_FILE = PROCESSED_DATA / "posts.csv"
EDGES_FILE = PROCESSED_DATA / "edges.csv"
CLAIMS_FILE = PROCESSED_DATA / "claims.csv"
TEMPORAL_FILE = PROCESSED_DATA / "temporal_features.csv"
TREES_FILE = PROCESSED_DATA / "propagation_trees.json"


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 60)
print("PHEME DATASET VALIDATION")
print("=" * 60)

posts = pd.read_csv(POSTS_FILE)
edges = pd.read_csv(EDGES_FILE)
claims = pd.read_csv(CLAIMS_FILE)
temporal = pd.read_csv(TEMPORAL_FILE)

with open(TREES_FILE, "r", encoding="utf-8") as file:
    trees = json.load(file)


# ============================================================
# BASIC COUNTS
# ============================================================

print("\nBASIC COUNTS")
print("-" * 60)

print(f"Posts              : {len(posts)}")
print(f"Edges              : {len(edges)}")
print(f"Claims             : {len(claims)}")
print(f"Temporal threads   : {len(temporal)}")
print(f"Propagation trees  : {len(trees)}")


# ============================================================
# THREAD COUNTS
# ============================================================

post_threads = set(posts["root_id"].astype(str))
edge_threads = set(edges["root_id"].astype(str))
claim_threads = set(claims["root_id"].astype(str))
temporal_threads = set(temporal["root_id"].astype(str))

tree_threads = set(
    str(tree["root_id"])
    for tree in trees
)


print("\nTHREAD COVERAGE")
print("-" * 60)

print(f"Threads in posts       : {len(post_threads)}")
print(f"Threads in edges       : {len(edge_threads)}")
print(f"Threads in claims      : {len(claim_threads)}")
print(f"Threads in temporal    : {len(temporal_threads)}")
print(f"Threads in trees       : {len(tree_threads)}")


# ============================================================
# THREADS WITHOUT EDGES
# ============================================================

threads_without_edges = (
    post_threads - edge_threads
)

print("\nTHREADS WITHOUT EDGES")
print("-" * 60)

print(
    f"Threads without edges : "
    f"{len(threads_without_edges)}"
)


# ============================================================
# POSTS WITHOUT PROPAGATION TREE
# ============================================================

posts_without_tree = (
    post_threads - tree_threads
)

print("\nPROPAGATION COVERAGE")
print("-" * 60)

print(
    f"Threads without tree  : "
    f"{len(posts_without_tree)}"
)


# ============================================================
# INVALID EDGE REFERENCES
# ============================================================

post_ids = set(
    posts["post_id"].astype(str)
)

edge_parent_ids = set(
    edges["parent_id"].astype(str)
)

edge_child_ids = set(
    edges["child_id"].astype(str)
)

missing_parents = (
    edge_parent_ids - post_ids
)

missing_children = (
    edge_child_ids - post_ids
)

print("\nEDGE VALIDATION")
print("-" * 60)

print(
    f"Edges with missing parent : "
    f"{len(missing_parents)}"
)

print(
    f"Edges with missing child  : "
    f"{len(missing_children)}"
)


# ============================================================
# DUPLICATE POSTS
# ============================================================

duplicate_posts = posts[
    posts["post_id"].duplicated()
]

print("\nDUPLICATE CHECK")
print("-" * 60)

print(
    f"Duplicate post IDs : "
    f"{len(duplicate_posts)}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLABEL DISTRIBUTION")
print("-" * 60)

print(
    claims["label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# UNKNOWN LABELS
# ============================================================

unknown_claims = claims[
    claims["label"] == "unknown"
]

print("\nUNKNOWN LABELS")
print("-" * 60)

print(
    f"Unknown claims : "
    f"{len(unknown_claims)}"
)

if len(unknown_claims) > 0:
    print(
        unknown_claims[
            [
                "root_id",
                "event",
                "thread_type",
                "is_rumour"
            ]
        ].to_string(index=False)
    )


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

print("\nTEMPORAL VALIDATION")
print("-" * 60)

print(
    f"Missing duration values : "
    f"{temporal['duration_seconds'].isna().sum()}"
)

print(
    f"Missing response times  : "
    f"{temporal['response_time_seconds'].isna().sum()}"
)

print(
    f"Missing depths          : "
    f"{temporal['conversation_depth'].isna().sum()}"
)

print(
    f"Missing branching       : "
    f"{temporal['max_branching_factor'].isna().sum()}"
)


# ============================================================
# NEGATIVE VALUES
# ============================================================

negative_duration = (
    temporal["duration_seconds"] < 0
).sum()

negative_response = (
    temporal["response_time_seconds"] < 0
).sum()

print(
    f"Negative durations      : "
    f"{negative_duration}"
)

print(
    f"Negative response times : "
    f"{negative_response}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)