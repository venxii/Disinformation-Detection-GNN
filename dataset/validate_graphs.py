import pandas as pd
import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"

METADATA_FILE = GRAPHS_DIR / "graph_metadata.csv"
CLAIMS_FILE = PROJECT_ROOT / "data" / "processed" / "claims.csv"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("GNN GRAPH DATASET VALIDATION")
print("=" * 60)


# ============================================================
# LOAD METADATA AND CLAIMS
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
claims = pd.read_csv(CLAIMS_FILE)

metadata["root_id"] = metadata["root_id"].astype(str)
claims["root_id"] = claims["root_id"].astype(str)

print("\nMETADATA")
print("-" * 60)

print(f"Metadata rows : {len(metadata)}")
print(f"Claims rows   : {len(claims)}")


# ============================================================
# FIND GRAPH FILES
# ============================================================

graph_files = sorted(
    GRAPHS_DIR.glob("graph_*.json")
)

print("\nGRAPH FILES")
print("-" * 60)

print(
    f"Graph JSON files found : "
    f"{len(graph_files)}"
)


# ============================================================
# VALIDATION COUNTERS
# ============================================================

valid_graphs = 0
invalid_json = 0
missing_labels = 0
missing_nodes = 0
missing_edges = 0
invalid_edges = 0
count_mismatches = 0
label_mismatches = 0


# ============================================================
# VALIDATE EACH GRAPH
# ============================================================

print("\nValidating graphs...")

for index, graph_file in enumerate(
    graph_files,
    start=1
):

    try:

        with open(
            graph_file,
            "r",
            encoding="utf-8"
        ) as file:

            graph = json.load(file)

    except Exception as error:

        print(
            f"INVALID JSON: "
            f"{graph_file.name} -> {error}"
        )

        invalid_json += 1
        continue


    # --------------------------------------------------------
    # BASIC GRAPH FIELDS
    # --------------------------------------------------------

    root_id = str(
        graph.get("root_id", "")
    )

    label = graph.get("label")

    nodes = graph.get("nodes")

    edges = graph.get("edges")


    # --------------------------------------------------------
    # LABEL CHECK
    # --------------------------------------------------------

    if label is None or str(label).strip() == "":
        missing_labels += 1


    # --------------------------------------------------------
    # NODE CHECK
    # --------------------------------------------------------

    if not isinstance(nodes, list):
        missing_nodes += 1
        nodes = []


    # --------------------------------------------------------
    # EDGE CHECK
    # --------------------------------------------------------

    if not isinstance(edges, list):
        missing_edges += 1
        edges = []


    # --------------------------------------------------------
    # NODE IDs
    # --------------------------------------------------------

    node_ids = set()

    for node in nodes:

        if not isinstance(node, dict):
            continue

        post_id = node.get("post_id")

        if post_id is not None:
            node_ids.add(str(post_id))


    # --------------------------------------------------------
    # EDGE REFERENCES
    # --------------------------------------------------------

    graph_has_invalid_edge = False

    for edge in edges:

        if not isinstance(edge, dict):
            graph_has_invalid_edge = True
            continue

        source = str(
            edge.get("source", "")
        )

        target = str(
            edge.get("target", "")
        )

        if (
            source not in node_ids
            or target not in node_ids
        ):

            graph_has_invalid_edge = True


    if graph_has_invalid_edge:
        invalid_edges += 1


    # --------------------------------------------------------
    # NODE/EDGE COUNT CHECK
    # --------------------------------------------------------

    declared_nodes = graph.get(
        "num_nodes",
        -1
    )

    declared_edges = graph.get(
        "num_edges",
        -1
    )

    if (
        declared_nodes != len(nodes)
        or declared_edges != len(edges)
    ):

        count_mismatches += 1


    # --------------------------------------------------------
    # LABEL CROSS-CHECK
    # --------------------------------------------------------

    claim_rows = claims[
        claims["root_id"] == root_id
    ]

    if len(claim_rows) == 1:

        expected_label = str(
            claim_rows.iloc[0]["label"]
        )

        actual_label = str(label)

        if expected_label != actual_label:
            label_mismatches += 1


    valid_graphs += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 500 == 0:

        print(
            f"Validated "
            f"{index}/{len(graph_files)} graphs"
        )


# ============================================================
# METADATA CHECK
# ============================================================

metadata_graph_ids = set(
    metadata["graph_id"].astype(str)
)

file_graph_ids = set(
    file.name
    for file in graph_files
)

missing_from_metadata = (
    file_graph_ids - metadata_graph_ids
)

missing_files = (
    metadata_graph_ids - file_graph_ids
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("GRAPH VALIDATION RESULTS")
print("=" * 60)

print(
    f"Graph files found       : {len(graph_files)}"
)

print(
    f"Valid JSON graphs       : {valid_graphs}"
)

print(
    f"Invalid JSON files      : {invalid_json}"
)

print(
    f"Graphs missing labels   : {missing_labels}"
)

print(
    f"Graphs missing nodes    : {missing_nodes}"
)

print(
    f"Graphs missing edges    : {missing_edges}"
)

print(
    f"Graphs with invalid edges : {invalid_edges}"
)

print(
    f"Node/edge count errors  : {count_mismatches}"
)

print(
    f"Label mismatches        : {label_mismatches}"
)

print(
    f"Metadata missing graphs : {len(missing_from_metadata)}"
)

print(
    f"Missing graph files     : {len(missing_files)}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("GRAPH LABEL DISTRIBUTION")
print("-" * 60)

print(
    metadata["label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 60)

if (
    len(graph_files) == len(metadata)
    and invalid_json == 0
    and missing_labels == 0
    and missing_nodes == 0
    and missing_edges == 0
    and invalid_edges == 0
    and count_mismatches == 0
    and label_mismatches == 0
    and len(missing_from_metadata) == 0
    and len(missing_files) == 0
):

    print("GRAPH DATASET VALIDATION PASSED")
    print("Your GNN-ready graph dataset is consistent.")

else:

    print("GRAPH DATASET VALIDATION FOUND ISSUES")
    print("Review the values above before handoff.")

print("=" * 60)