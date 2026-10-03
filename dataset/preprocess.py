import json
import csv
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "pheme" / "all-rnr-annotated-threads"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

POSTS_FILE = PROCESSED_DATA / "posts.csv"
CLAIMS_FILE = PROCESSED_DATA / "claims.csv"
EDGES_FILE = PROCESSED_DATA / "edges.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"Could not read {file_path}: {error}")
        return None


def find_json_files(folder):
    """
    Find JSON files while ignoring macOS metadata files
    such as ._filename.
    """
    if not folder.exists():
        return []

    return [
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file()
        and not file_path.name.startswith("._")
        and file_path.name != ".DS_Store"
    ]


def get_veracity_label(annotation):
    if not annotation:
        return "unknown"

    is_rumour = str(
        annotation.get("is_rumour", "")
    ).strip().lower()

    misinformation = str(
        annotation.get("misinformation", "")
    ).strip().lower()

    true_value = str(
        annotation.get("true", "")
    ).strip().lower()

    # Non-rumour
    if is_rumour in ["nonrumour", "non-rumour"]:
        return "non-rumour"

    # Rumour
    if is_rumour == "rumour":

        if true_value == "1":
            return "true"

        if misinformation == "1":
            return "false"

        return "unverified"

    return "unknown"


def flatten_structure(tree, parent_id=None, edges=None):
    """
    Convert PHEME's nested structure JSON into
    parent-child edges.

    Example:

    {
        "100": {
            "101": {
                "102": {}
            }
        }
    }

    becomes:

    100 -> 101
    101 -> 102
    """

    if edges is None:
        edges = []

    if not isinstance(tree, dict):
        return edges

    for node_id, children in tree.items():

        if parent_id is not None:
            edges.append({
                "parent_id": str(parent_id),
                "child_id": str(node_id)
            })

        flatten_structure(
            children,
            parent_id=node_id,
            edges=edges
        )

    return edges


# ============================================================
# PROCESS ONE THREAD
# ============================================================

def process_thread(thread_path, event_name, thread_type):
    """
    Process one PHEME conversation thread.
    """

    thread_id = thread_path.name

    
    structure_path = thread_path / "structure.json"

    source_folder = thread_path / "source-tweets"
    reactions_folder = thread_path / "reactions"

        # --------------------------------------------------------
        # Read annotation
        # --------------------------------------------------------

    annotation_path = thread_path / "annotation.json"

    # --------------------------------------------------------
    # Read annotation
    # --------------------------------------------------------

    annotation = load_json(annotation_path)

    # PHEME non-rumours may not contain an annotation file.
    # Their thread_type itself tells us the label.
    if annotation is None:

        if thread_type == "non-rumours":
            annotation = {
                "is_rumour": "non-rumour",
                "misinformation": 0,
                "true": 0,
                "is_turnaround": 0,
                "category": ""
            }
        else:
            print(f"Skipping {thread_id}: annotation missing")
            return [], [], []

    label = get_veracity_label(annotation)

    # --------------------------------------------------------
    # Read source tweet
    # --------------------------------------------------------

    posts = []

    source_files = find_json_files(source_folder)

    for source_file in source_files:

        tweet = load_json(source_file)

        if not isinstance(tweet, dict):
            continue

        post_id = str(tweet.get("id"))

        posts.append({
            "post_id": post_id,
            "root_id": thread_id,
            "parent_id": "",
            "text": tweet.get("text", ""),
            "timestamp": tweet.get("created_at", ""),
            "post_type": "source",
            "event": event_name,
            "thread_type": thread_type
        })

    # --------------------------------------------------------
    # Read reactions
    # --------------------------------------------------------

    reaction_files = find_json_files(reactions_folder)

    for reaction_file in reaction_files:

        tweet = load_json(reaction_file)

        if not isinstance(tweet, dict):
            continue

        post_id = str(tweet.get("id"))

        parent_id = tweet.get("in_reply_to_status_id")

        if parent_id is not None:
            parent_id = str(parent_id)
        else:
            parent_id = ""

        posts.append({
            "post_id": post_id,
            "root_id": thread_id,
            "parent_id": parent_id,
            "text": tweet.get("text", ""),
            "timestamp": tweet.get("created_at", ""),
            "post_type": "reaction",
            "event": event_name,
            "thread_type": thread_type
        })

    # --------------------------------------------------------
    # Read structure and create graph edges
    # --------------------------------------------------------

    edges = []

    structure = load_json(structure_path)

    if structure is not None:

        edges = flatten_structure(
            structure,
            parent_id=None
        )

        # Add root/thread information
        for edge in edges:
            edge["root_id"] = thread_id
            edge["event"] = event_name

    # --------------------------------------------------------
    # Create claim record
    # --------------------------------------------------------

    claims = [{
        "root_id": thread_id,
        "event": event_name,
        "thread_type": thread_type,
        "label": label,
        "is_rumour": annotation.get("is_rumour", ""),
        "misinformation": annotation.get("misinformation", ""),
        "true": annotation.get("true", ""),
        "is_turnaround": annotation.get("is_turnaround", ""),
        "category": annotation.get("category", "")
    }]

    return posts, claims, edges


# ============================================================
# MAIN DATASET PROCESSING
# ============================================================

def main():

    print("=" * 60)
    print("PHEME DATA PREPROCESSING")
    print("=" * 60)

    if not RAW_DATA.exists():

        print("\nERROR: PHEME dataset was not found.")
        print(f"Expected location:\n{RAW_DATA}")
        return

    PROCESSED_DATA.mkdir(
        parents=True,
        exist_ok=True
    )

    all_posts = []
    all_claims = []
    all_edges = []

    thread_count = 0

    # --------------------------------------------------------
    # Process each event
    # --------------------------------------------------------

    event_folders = [
        folder
        for folder in RAW_DATA.iterdir()
        if folder.is_dir()
        and not folder.name.startswith(".")
    ]

    print(f"\nEvents found: {len(event_folders)}")

    for event_folder in sorted(event_folders):

        event_name = event_folder.name

        print(f"\nProcessing event: {event_name}")

        # ----------------------------------------------------
        # Process rumours and non-rumours
        # ----------------------------------------------------

        for thread_type in ["rumours", "non-rumours"]:

            type_folder = event_folder / thread_type

            if not type_folder.exists():
                continue

            thread_folders = [
                folder
                for folder in type_folder.iterdir()
                if folder.is_dir()
                and not folder.name.startswith(".")
            ]

            print(
                f"  {thread_type}: "
                f"{len(thread_folders)} threads"
            )

            for thread_folder in thread_folders:

                posts, claims, edges = process_thread(
                    thread_folder,
                    event_name,
                    thread_type
                )

                all_posts.extend(posts)
                all_claims.extend(claims)
                all_edges.extend(edges)

                thread_count += 1

    # ========================================================
    # REMOVE DUPLICATE POSTS
    # ========================================================

    unique_posts = {}

    for post in all_posts:

        post_id = post["post_id"]

        if post_id not in unique_posts:
            unique_posts[post_id] = post

    all_posts = list(unique_posts.values())

    # ========================================================
    # WRITE POSTS.CSV
    # ========================================================

    with open(
        POSTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "post_id",
                "root_id",
                "parent_id",
                "text",
                "timestamp",
                "post_type",
                "event",
                "thread_type"
            ]
        )

        writer.writeheader()
        writer.writerows(all_posts)

    # ========================================================
    # WRITE CLAIMS.CSV
    # ========================================================

    with open(
        CLAIMS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "root_id",
                "event",
                "thread_type",
                "label",
                "is_rumour",
                "misinformation",
                "true",
                "is_turnaround",
                "category"
            ]
        )

        writer.writeheader()
        writer.writerows(all_claims)

    # ========================================================
    # WRITE EDGES.CSV
    # ========================================================

    with open(
        EDGES_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "root_id",
                "event",
                "parent_id",
                "child_id"
            ]
        )

        writer.writeheader()
        writer.writerows(all_edges)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Threads processed : {thread_count}")
    print(f"Posts extracted   : {len(all_posts)}")
    print(f"Claims extracted  : {len(all_claims)}")
    print(f"Edges extracted   : {len(all_edges)}")

    print("\nOutput files:")

    print(f"  {POSTS_FILE}")
    print(f"  {CLAIMS_FILE}")
    print(f"  {EDGES_FILE}")

    print("\nNext step:")
    print("Inspect the generated CSV files before building")
    print("propagation and temporal features.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()