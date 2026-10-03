"""
Integration check of nlp/embedding_node_helper.py on Person 1's graphs.

1. Coverage: every node of every graph JSON gets exactly one embedding.
2. Independent check on selected graphs: each node's own "text" field
   (from the graph JSON) is normalised and re-embedded with BERTweet;
   the result must match the row returned by the helper.
3. Order check: shuffling a graph's node list shuffles the rows identically.

Graph files are only read, never modified.

Run from the project root:
    .venv/bin/python -m nlp.check_node_helper
"""

import json
import resource
import sys
import time

import numpy as np

from nlp.bertweet_embeddings import embed_texts, load_bertweet, select_device
from nlp.config import EMBEDDING_DIM, GRAPHS_DIR, NLP_DATA
from nlp.embedding_node_helper import (
    get_graph_node_embeddings,
    get_node_embeddings,
    load_embedding_store,
    load_graph,
)
from nlp.text_preprocessing import normalize_for_bertweet


REPORT_FILE = NLP_DATA / "node_helper_check.json"

# Largest graph, a single-node graph, a multi-node graph without edges
FIXED_GRAPHS = ["graph_003839.json", "graph_000021.json", "graph_002129.json"]
RANDOM_GRAPHS = 3
SEED = 0


def coverage_check(store, graph_files):

    total_nodes = 0
    graphs_with_problems = []
    seen = {}

    start = time.time()

    for path in graph_files:

        try:
            node_ids, features = get_graph_node_embeddings(path, store)
        except Exception as error:
            graphs_with_problems.append({"graph": path.name, "error": str(error)})
            continue

        if features.shape != (len(node_ids), EMBEDDING_DIM):
            graphs_with_problems.append({"graph": path.name, "error": "shape"})

        for post_id in node_ids:
            seen[post_id] = seen.get(post_id, 0) + 1

        total_nodes += len(node_ids)

    return {
        "graphs": len(graph_files),
        "graphs_with_problems": graphs_with_problems,
        "total_nodes": total_nodes,
        "distinct_post_ids": len(seen),
        "post_ids_in_more_than_one_graph": sum(1 for c in seen.values() if c > 1),
        "embedding_rows_not_used_by_any_graph": len(store) - len(set(seen) & set(store.index)),
        "seconds": round(time.time() - start, 1),
    }


def independent_check(store, graph_files, tokenizer, model, device):

    results = []

    for path in graph_files:

        graph = load_graph(path)
        node_ids, features = get_graph_node_embeddings(graph, store)

        texts = [normalize_for_bertweet(node["text"]) for node in graph["nodes"]]
        recomputed = embed_texts(texts, tokenizer, model, device, sort_by_length=False)

        per_node = np.abs(recomputed - features).max(axis=1)

        results.append({
            "graph": path.name,
            "label": graph["label"],
            "num_nodes": len(node_ids),
            "num_edges": len(graph["edges"]),
            "max_abs_diff": float(per_node.max()),
            "nodes_within_1e-3": int((per_node < 1e-3).sum()),
        })

    return results


def order_check(store, graph_file):

    node_ids, features = get_graph_node_embeddings(graph_file, store)
    permutation = np.random.default_rng(SEED).permutation(len(node_ids))
    shuffled = get_node_embeddings([node_ids[i] for i in permutation], store)

    return {
        "graph": graph_file.name,
        "num_nodes": len(node_ids),
        "rows_follow_shuffled_order": bool(np.array_equal(shuffled, features[permutation])),
    }


def main():

    print("=" * 60)
    print("NODE HELPER INTEGRATION CHECK")
    print("=" * 60)

    store = load_embedding_store(mmap=True)
    print(f"\nEmbedding store : {len(store)} posts x {store.dim} "
          f"({type(store.embeddings).__name__})")

    graph_files = sorted(GRAPHS_DIR.glob("graph_*.json"))

    coverage = coverage_check(store, graph_files)
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    print(f"\nCoverage over {coverage['graphs']} graphs ({coverage['seconds']} s):")
    for key in ["total_nodes", "distinct_post_ids", "post_ids_in_more_than_one_graph",
                "embedding_rows_not_used_by_any_graph"]:
        print(f"  {key:<38}: {coverage[key]}")
    print(f"  graphs with problems                  : {len(coverage['graphs_with_problems'])}")
    print(f"  peak process memory (MB)              : {max_rss_mb:.0f}")

    rng = np.random.default_rng(SEED)
    others = [p for p in graph_files if p.name not in FIXED_GRAPHS]
    chosen = [GRAPHS_DIR / name for name in FIXED_GRAPHS] + [
        others[i] for i in sorted(rng.choice(len(others), RANDOM_GRAPHS, replace=False))
    ]

    device = select_device("auto")
    tokenizer, model = load_bertweet(device)

    independent = independent_check(store, chosen, tokenizer, model, device)

    print(f"\nIndependent re-embedding check (device {device}):")
    for result in independent:
        print(f"  {result['graph']}  nodes={result['num_nodes']:<4} "
              f"edges={result['num_edges']:<4} "
              f"within 1e-3: {result['nodes_within_1e-3']}/{result['num_nodes']}  "
              f"max diff {result['max_abs_diff']:.1e}")

    order = order_check(store, GRAPHS_DIR / FIXED_GRAPHS[0])
    print(f"\nShuffled order check on {order['graph']}: "
          f"{'OK' if order['rows_follow_shuffled_order'] else 'FAILED'}")

    passed = (
        not coverage["graphs_with_problems"]
        and coverage["total_nodes"] == len(store)
        and coverage["post_ids_in_more_than_one_graph"] == 0
        and coverage["embedding_rows_not_used_by_any_graph"] == 0
        and all(r["nodes_within_1e-3"] == r["num_nodes"] for r in independent)
        and order["rows_follow_shuffled_order"]
    )

    report = {
        "passed": passed,
        "store_rows": len(store),
        "memory_mapped": type(store.embeddings).__name__ == "memmap",
        "peak_process_memory_mb": round(max_rss_mb),
        "coverage": coverage,
        "independent_check_device": device,
        "independent_check": independent,
        "order_check": order,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved to: {REPORT_FILE}")
    print("\n" + "=" * 60)
    print("NODE HELPER CHECK PASSED" if passed else "NODE HELPER CHECK FAILED")
    print("=" * 60)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
