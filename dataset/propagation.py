import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
import json


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

POSTS_FILE = PROCESSED_DIR / "posts.csv"
EDGES_FILE = PROCESSED_DIR / "edges.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("PHEME PROPAGATION TREE CONSTRUCTION")
    print("=" * 60)

    print("\nLoading posts.csv...")
    posts = pd.read_csv(POSTS_FILE)

    print("Loading edges.csv...")
    edges = pd.read_csv(EDGES_FILE)

    print(f"Posts loaded : {len(posts)}")
    print(f"Edges loaded : {len(edges)}")

    return posts, edges


# ============================================================
# BUILD ONE PROPAGATION TREE
# ============================================================

def build_tree(root_id, group):
    """
    Build a propagation tree for one PHEME conversation.

    Each edge represents:

        parent_id -> child_id
    """

    children = defaultdict(list)

    for _, row in group.iterrows():
        parent = str(row["parent_id"])
        child = str(row["child_id"])

        children[parent].append(child)

    root = str(root_id)

    # Nodes discovered in the tree
    nodes = set()

    # Store parent -> child relationships
    tree_edges = []

    # BFS traversal
    queue = deque()

    queue.append((root, 0))
    visited = set()

    max_depth = 0

    while queue:

        node, depth = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        nodes.add(node)

        max_depth = max(max_depth, depth)

        for child in children.get(node, []):

            tree_edges.append({
                "parent_id": node,
                "child_id": child
            })

            queue.append((child, depth + 1))

    return {
        "root_id": root,
        "nodes": list(nodes),
        "edges": tree_edges,
        "num_nodes": len(nodes),
        "num_edges": len(tree_edges),
        "depth": max_depth
    }


# ============================================================
# BUILD ALL PROPAGATION TREES
# ============================================================

def build_all_trees(edges):

    trees = []

    grouped = edges.groupby("root_id")

    total = len(grouped)

    print(f"\nBuilding propagation trees for {total} threads...\n")

    for index, (root_id, group) in enumerate(grouped, start=1):

        tree = build_tree(root_id, group)

        trees.append(tree)

        if index % 500 == 0 or index == total:
            print(
                f"Processed {index}/{total} threads"
            )

    return trees


# ============================================================
# SAVE TREES
# ============================================================

def save_trees(trees):

    output_file = OUTPUT_DIR / "propagation_trees.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            trees,
            f,
            indent=2
        )

    print("\nPropagation trees saved to:")
    print(output_file)


# ============================================================
# GENERATE STATISTICS
# ============================================================

def generate_statistics(trees):

    if not trees:
        print("No propagation trees found.")
        return

    total_nodes = sum(
        tree["num_nodes"]
        for tree in trees
    )

    total_edges = sum(
        tree["num_edges"]
        for tree in trees
    )

    depths = [
        tree["depth"]
        for tree in trees
    ]

    print("\n" + "=" * 60)
    print("PROPAGATION STATISTICS")
    print("=" * 60)

    print(f"Threads          : {len(trees)}")
    print(f"Total nodes      : {total_nodes}")
    print(f"Total edges      : {total_edges}")
    print(f"Maximum depth    : {max(depths)}")
    print(
        f"Average depth    : "
        f"{sum(depths) / len(depths):.2f}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    posts, edges = load_data()

    trees = build_all_trees(edges)

    save_trees(trees)

    generate_statistics(trees)

    print("\nPropagation tree construction complete.")


if __name__ == "__main__":
    main()