import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"

POSTS_FILE = PROCESSED_DATA / "posts.csv"
EDGES_FILE = PROCESSED_DATA / "edges.csv"
CLAIMS_FILE = PROCESSED_DATA / "claims.csv"
TEMPORAL_FILE = PROCESSED_DATA / "temporal_features.csv"

METADATA_FILE = GRAPHS_DIR / "graph_metadata.csv"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

GRAPHS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PHEME GRAPH EXPORT")
print("=" * 60)

print("\nLoading datasets...")

posts = pd.read_csv(POSTS_FILE)
edges = pd.read_csv(EDGES_FILE)
claims = pd.read_csv(CLAIMS_FILE)
temporal = pd.read_csv(TEMPORAL_FILE)

print(f"Posts loaded   : {len(posts)}")
print(f"Edges loaded   : {len(edges)}")
print(f"Claims loaded  : {len(claims)}")
print(f"Temporal rows  : {len(temporal)}")


# ============================================================
# NORMALIZE IDs
# ============================================================

posts["root_id"] = posts["root_id"].astype(str)
posts["post_id"] = posts["post_id"].astype(str)

edges["root_id"] = edges["root_id"].astype(str)
edges["parent_id"] = edges["parent_id"].astype(str)
edges["child_id"] = edges["child_id"].astype(str)

claims["root_id"] = claims["root_id"].astype(str)
temporal["root_id"] = temporal["root_id"].astype(str)


# ============================================================
# INDEX DATA
# ============================================================

print("\nIndexing data...")

posts_by_thread = {
    root_id: group
    for root_id, group in posts.groupby("root_id")
}

edges_by_thread = {
    root_id: group
    for root_id, group in edges.groupby("root_id")
}

claims_by_thread = {
    root_id: row
    for root_id, row in claims.set_index("root_id").iterrows()
}

temporal_by_thread = {
    root_id: row
    for root_id, row in temporal.set_index("root_id").iterrows()
}


# ============================================================
# HELPER: SAFE VALUE
# ============================================================

def safe_value(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


# ============================================================
# EXPORT GRAPHS
# ============================================================

print("\nExporting graphs...")

metadata = []

total_threads = len(posts_by_thread)

exported = 0
single_node_graphs = 0
skipped_edges = 0


for index, (root_id, thread_posts) in enumerate(
    posts_by_thread.items(),
    start=1
):

    # --------------------------------------------------------
    # CLAIM INFORMATION
    # --------------------------------------------------------

    claim = claims_by_thread.get(root_id)

    if claim is None:
        print(
            f"WARNING: No claim found for thread {root_id}"
        )

        continue

    label = safe_value(claim.get("label"))
    event = safe_value(claim.get("event"))
    thread_type = safe_value(
        claim.get("thread_type")
    )

    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    temporal_row = temporal_by_thread.get(root_id)

    temporal_features = {}

    if temporal_row is not None:

        temporal_columns = [
            "duration_seconds",
            "num_posts",
            "num_source_posts",
            "num_reactions",
            "response_time_seconds",
            "average_response_time_seconds",
            "propagation_speed",
            "posts_per_minute",
            "conversation_depth",
            "max_branching_factor",
            "average_branching_factor"
        ]

        for column in temporal_columns:

            if column in temporal_row.index:

                temporal_features[column] = safe_value(
                    temporal_row[column]
                )

    # --------------------------------------------------------
    # CREATE NODE ID SET
    # --------------------------------------------------------

    valid_post_ids = set(
        thread_posts["post_id"].astype(str)
    )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    nodes = []

    for _, post in thread_posts.iterrows():

        node = {
            "post_id": safe_value(post["post_id"]),
            "text": safe_value(post.get("text", "")),
            "timestamp": safe_value(
                post.get("timestamp", "")
            ),
            "parent_id": safe_value(
                post.get("parent_id", "")
            ),
            "post_type": safe_value(
                post.get("post_type", "")
            )
        }

        nodes.append(node)

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    graph_edges = []

    thread_edges = edges_by_thread.get(
        root_id,
        pd.DataFrame()
    )

    for _, edge in thread_edges.iterrows():

        parent_id = str(edge["parent_id"])
        child_id = str(edge["child_id"])

        # Keep only edges whose child exists in the thread.
        if child_id not in valid_post_ids:

            skipped_edges += 1
            continue

        # Parent must also exist.
        if parent_id not in valid_post_ids:

            skipped_edges += 1
            continue

        graph_edges.append({
            "source": parent_id,
            "target": child_id
        })

    if len(graph_edges) == 0:
        single_node_graphs += 1

    # --------------------------------------------------------
    # GRAPH OBJECT
    # --------------------------------------------------------

    graph = {
        "root_id": root_id,

        "label": label,

        "event": event,

        "thread_type": thread_type,

        "num_nodes": len(nodes),

        "num_edges": len(graph_edges),

        "nodes": nodes,

        "edges": graph_edges,

        "temporal_features": temporal_features
    }

    # --------------------------------------------------------
    # FILE NAME
    # --------------------------------------------------------

    graph_number = index

    graph_filename = (
        f"graph_{graph_number:06d}.json"
    )

    graph_path = GRAPHS_DIR / graph_filename

    # --------------------------------------------------------
    # SAVE GRAPH
    # --------------------------------------------------------

    with open(
        graph_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            graph,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata.append({

        "graph_id":
            graph_filename,

        "root_id":
            root_id,

        "label":
            label,

        "event":
            event,

        "thread_type":
            thread_type,

        "num_nodes":
            len(nodes),

        "num_edges":
            len(graph_edges),

        "duration_seconds":
            temporal_features.get(
                "duration_seconds"
            ),

        "response_time_seconds":
            temporal_features.get(
                "response_time_seconds"
            ),

        "propagation_speed":
            temporal_features.get(
                "propagation_speed"
            ),

        "conversation_depth":
            temporal_features.get(
                "conversation_depth"
            ),

        "max_branching_factor":
            temporal_features.get(
                "max_branching_factor"
            ),

        "average_branching_factor":
            temporal_features.get(
                "average_branching_factor"
            )
    })

    exported += 1

    if index % 500 == 0:
        print(
            f"Exported {index}/{total_threads} graphs"
        )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("=" * 60)
print("GRAPH EXPORT COMPLETE")
print("=" * 60)

print(
    f"Graphs exported      : {exported}"
)

print(
    f"Single/no-edge graphs: {single_node_graphs}"
)

print(
    f"Skipped invalid edges: {skipped_edges}"
)

print(
    f"Metadata rows        : {len(metadata_df)}"
)

print()
print("Graphs saved to:")
print(GRAPHS_DIR)

print()
print("Metadata saved to:")
print(METADATA_FILE)

print("=" * 60)