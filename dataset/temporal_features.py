import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

POSTS_FILE = PROCESSED_DATA / "posts.csv"
TREES_FILE = PROCESSED_DATA / "propagation_trees.json"
OUTPUT_FILE = PROCESSED_DATA / "temporal_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading posts...")
posts = pd.read_csv(POSTS_FILE)

print("Loading propagation trees...")
with open(TREES_FILE, "r", encoding="utf-8") as file:
    propagation_trees = json.load(file)

# Convert list of trees into a dictionary using root_id
if isinstance(propagation_trees, list):
    propagation_trees = {
        str(tree["root_id"]): tree
        for tree in propagation_trees
    }

print(f"Posts loaded : {len(posts)}")
print(f"Propagation trees loaded : {len(propagation_trees)}")


# ============================================================
# PREPARE DATA
# ============================================================

posts["root_id"] = posts["root_id"].astype(str)

print("Converting timestamps...")

posts["timestamp"] = pd.to_datetime(
    posts["timestamp"],
    errors="coerce",
    utc=True
)

invalid_timestamps = posts["timestamp"].isna().sum()

print(f"Invalid timestamps : {invalid_timestamps}")


# ============================================================
# TREE STATISTICS
# ============================================================

def calculate_tree_features(tree):

    edges = tree.get("edges", [])

    children = defaultdict(list)

    for edge in edges:
        parent = str(edge["parent_id"])
        child = str(edge["child_id"])

        children[parent].append(child)

    # Number of children for each parent
    branching_counts = [
        len(child_list)
        for child_list in children.values()
    ]

    if branching_counts:
        max_branching_factor = max(branching_counts)

        average_branching_factor = (
            sum(branching_counts) / len(branching_counts)
        )
    else:
        max_branching_factor = 0
        average_branching_factor = 0

    conversation_depth = tree.get("depth", 0)

    return (
        conversation_depth,
        max_branching_factor,
        average_branching_factor
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

print("Extracting temporal and propagation features...")

results = []

grouped_posts = posts.groupby("root_id")

total_threads = len(grouped_posts)

for count, (root_id, thread) in enumerate(grouped_posts, start=1):

    # --------------------------------------------------------
    # Sort posts by timestamp
    # --------------------------------------------------------

    thread = thread.sort_values("timestamp")

    valid_times = thread["timestamp"].dropna()

    if len(valid_times) == 0:
        continue

    first_timestamp = valid_times.min()
    last_timestamp = valid_times.max()

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration_seconds = (
        last_timestamp - first_timestamp
    ).total_seconds()

    if duration_seconds < 0:
        duration_seconds = 0

    # --------------------------------------------------------
    # Basic post statistics
    # --------------------------------------------------------

    num_posts = len(thread)

    num_source_posts = (
        thread["post_type"] == "source"
    ).sum()

    num_reactions = (
        thread["post_type"] == "reaction"
    ).sum()

    # --------------------------------------------------------
    # RESPONSE TIME
    # --------------------------------------------------------

    source_posts = thread[
        thread["post_type"] == "source"
    ]

    reaction_posts = thread[
        thread["post_type"] == "reaction"
    ]

    if len(source_posts) > 0 and len(reaction_posts) > 0:

        root_timestamp = source_posts["timestamp"].min()

        reaction_times = reaction_posts["timestamp"].dropna()

        if len(reaction_times) > 0:

            first_reaction_time = reaction_times.min()

            response_time_seconds = (
                first_reaction_time - root_timestamp
            ).total_seconds()

            if response_time_seconds < 0:
                response_time_seconds = 0

            # Average time taken for reactions
            reaction_delays = (
                reaction_times - root_timestamp
            ).dt.total_seconds()

            average_response_time_seconds = (
                reaction_delays.mean()
            )

        else:
            response_time_seconds = 0
            average_response_time_seconds = 0

    else:
        response_time_seconds = 0
        average_response_time_seconds = 0

    # --------------------------------------------------------
    # PROPAGATION SPEED
    # --------------------------------------------------------

    if duration_seconds > 0:

        propagation_speed = (
            num_reactions / duration_seconds
        )

        posts_per_minute = (
            num_posts / (duration_seconds / 60)
        )

    else:

        propagation_speed = 0
        posts_per_minute = 0

    # --------------------------------------------------------
    # PROPAGATION TREE FEATURES
    # --------------------------------------------------------

    tree = propagation_trees.get(str(root_id), {})

    (
        conversation_depth,
        max_branching_factor,
        average_branching_factor
    ) = calculate_tree_features(tree)

    # --------------------------------------------------------
    # STORE FEATURES
    # --------------------------------------------------------

    results.append({

        "root_id": root_id,

        "first_timestamp":
            first_timestamp.isoformat(),

        "last_timestamp":
            last_timestamp.isoformat(),

        "duration_seconds":
            duration_seconds,

        "num_posts":
            num_posts,

        "num_source_posts":
            num_source_posts,

        "num_reactions":
            num_reactions,

        "response_time_seconds":
            response_time_seconds,

        "average_response_time_seconds":
            average_response_time_seconds,

        "propagation_speed":
            propagation_speed,

        "posts_per_minute":
            posts_per_minute,

        "conversation_depth":
            conversation_depth,

        "max_branching_factor":
            max_branching_factor,

        "average_branching_factor":
            average_branching_factor
    })

    if count % 500 == 0:
        print(
            f"Processed {count}/{total_threads} threads"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

features_df = pd.DataFrame(results)


# ============================================================
# SAVE
# ============================================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("TEMPORAL + PROPAGATION FEATURES COMPLETE")
print("=" * 60)

print(f"Threads : {len(features_df)}")

print(
    f"Average duration : "
    f"{features_df['duration_seconds'].mean():.2f} seconds"
)

print(
    f"Average posts : "
    f"{features_df['num_posts'].mean():.2f}"
)

print(
    f"Average reactions : "
    f"{features_df['num_reactions'].mean():.2f}"
)

print(
    f"Average response time : "
    f"{features_df['response_time_seconds'].mean():.2f} seconds"
)

print(
    f"Average conversation depth : "
    f"{features_df['conversation_depth'].mean():.2f}"
)

print(
    f"Maximum conversation depth : "
    f"{features_df['conversation_depth'].max()}"
)

print(
    f"Maximum branching factor : "
    f"{features_df['max_branching_factor'].max()}"
)

print()
print("Output saved to:")
print(OUTPUT_FILE)

print("=" * 60)