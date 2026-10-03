# Dataset and Propagation Pipeline

## 1. Overview

This document describes the dataset preparation, preprocessing,
propagation graph construction, temporal feature extraction, graph
export and validation performed for the Disinformation Detection Using
Graph Neural Networks project.

The purpose of this pipeline is to transform the raw PHEME rumour
dataset into graph-structured data that can be used by the GNN
component of the project.

                    PHEME
                      │
                      ▼
              DATA PREPROCESSING
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Posts       Claims      Edges
          │           │           │
          └───────────┼───────────┘
                      ▼
             PROPAGATION TREES
                      │
                      ▼
            TEMPORAL FEATURES
                      │
                      ▼
                GRAPH EXPORT
                      │
                      ▼
              GRAPH VALIDATION
                      │
                      ▼
             ┌────────────────┐
             │ 6,425 GRAPHS   │
             │ GNN-READY      │
             └────────────────┘
                      │
                      ▼
                GNN TEAM

---

## 2. Dataset

Dataset used:

PHEME Dataset for Rumour Detection and Veracity Classification.

The dataset contains Twitter conversation threads collected around
different real-world events.

The processed dataset contains 6,425 conversation threads.

The events included are:

- Charlie Hebdo
- Ebola Essien
- Ferguson
- Germanwings Crash
- Gurlitt
- Ottawa Shooting
- Prince Toronto
- Putin Missing
- Sydney Siege

---

## 3. Dataset Organization

The raw PHEME dataset is stored under:

data/raw/pheme/all-rnr-annotated-threads/

The dataset is organized into event folders.

Each conversation thread contains information such as:

- Source tweet
- Reaction/reply tweets
- Conversation structure
- Rumour/veracity annotation

---

## 4. Preprocessing

Script:

dataset/preprocess.py

The preprocessing stage extracts information from the raw PHEME
conversation threads.

Three main CSV files are generated.

### posts.csv

Contains individual posts/tweets.

Important fields:

- post_id
- root_id
- parent_id
- text
- timestamp
- post_type
- event
- thread_type

### claims.csv

Contains thread-level annotation and labels.

Important fields:

- root_id
- event
- thread_type
- label
- is_rumour
- misinformation
- true
- is_turnaround
- category

### edges.csv

Contains reply relationships between posts.

Important fields:

- root_id
- event
- parent_id
- child_id

---

## 5. Label Processing

The PHEME annotations were converted into usable labels.

The final label distribution is:

| Label | Number |
|---|---:|
| non-rumour | 4022 |
| true | 1067 |
| unverified | 697 |
| false | 638 |
| unknown | 1 |

The single unknown record corresponds to a PHEME annotation whose
is_rumour value was "unclear". It was retained as unknown instead of
being assigned an unsupported label.

---

## 6. Dataset Statistics

After preprocessing:

- Threads: 6,425
- Posts: 104,582
- Extracted edges: 95,074

These files form the main processed dataset.

---

## 7. Propagation Graph Construction

Script:

dataset/propagation.py

The propagation structure of each conversation is represented as a
tree.

Each post is represented as a node.

A reply relationship is represented as a directed edge:

parent post -> child/reply post

For example:

Root Tweet
    |
    +---- Reply 1
    |       |
    |       +---- Reply 3
    |
    +---- Reply 2

The propagation tree information is stored in:

data/processed/propagation_trees.json

### Propagation statistics

- Threads with propagation trees: 5,741
- Total tree nodes: 100,656
- Total tree edges: 94,915
- Maximum propagation depth: 47
- Average propagation depth: 3.58

There are 684 threads without propagation edges. These are retained
during graph export as graphs containing their available post nodes
and zero edges.

---

## 8. Temporal and Propagation Features

Script:

dataset/temporal_features.py

Temporal and propagation characteristics were extracted for every
thread.

The generated file is:

data/processed/temporal_features.csv

The following features are included:

### Duration

The time between the first and last observed post in a conversation.

### Number of posts

Total number of posts in the conversation.

### Number of source posts

Number of source/root posts.

### Number of reactions

Number of reaction/reply posts.

### Response time

Time between the source post and the first reaction.

### Average response time

Average time from the source post to reaction posts.

### Propagation speed

Rate of reaction propagation over the conversation duration.

### Posts per minute

Number of posts relative to the conversation duration.

### Conversation depth

Maximum depth of the propagation tree.

### Maximum branching factor

Maximum number of direct child replies from a node.

### Average branching factor

Average number of children among nodes that have outgoing
propagation edges.

### Temporal statistics

- Threads: 6,425
- Average duration: 53,537.89 seconds
- Average posts: 16.28
- Average reactions: 15.28
- Average response time: 1,412.20 seconds
- Average conversation depth: 3.20
- Maximum conversation depth: 47
- Maximum branching factor: 122
- Invalid timestamps: 0

---

## 9. Dataset Validation

Script:

dataset/validate_dataset.py

The processed dataset was checked for structural consistency.

Validation results:

- Threads in posts: 6,425
- Threads in claims: 6,425
- Threads in temporal features: 6,425
- Threads with propagation trees: 5,741
- Threads without edges: 684
- Duplicate post IDs: 0
- Edges with missing parents: 0
- Edges with missing children in the original edge dataset: 1
- Missing temporal duration values: 0
- Missing response time values: 0
- Missing conversation depth values: 0
- Missing branching values: 0
- Negative durations: 0
- Negative response times: 0

The one invalid edge reference was safely excluded during graph export.

---

## 10. GNN Graph Export

Script:

dataset/graph_export.py

The processed information was converted into individual graph files.

Output directory:

data/graphs/

A graph represents one PHEME conversation thread.

Each graph contains:

### Graph-level information

- root_id
- label
- event
- thread_type
- number of nodes
- number of edges

### Node information

Each node represents a post and contains:

- post_id
- text
- timestamp
- parent_id
- post_type

### Edge information

Each edge represents a propagation/reply relationship:

source -> target

### Temporal information

Each graph also contains:

- duration_seconds
- num_posts
- num_source_posts
- num_reactions
- response_time_seconds
- average_response_time_seconds
- propagation_speed
- posts_per_minute
- conversation_depth
- max_branching_factor
- average_branching_factor

---

## 11. Final Graph Dataset

The graph export generated:

- 6,425 graph JSON files
- 6,425 metadata records
- 684 single/no-edge graphs
- 1 invalid original edge skipped during export

The metadata file is:

data/graphs/graph_metadata.csv

The graph files are:

graph_000001.json
graph_000002.json
...
graph_006425.json

---

## 12. Final Graph Validation

Script:

dataset/validate_graphs.py

The final graph dataset was automatically validated.

Results:

- Graph files found: 6,425
- Valid JSON graphs: 6,425
- Invalid JSON files: 0
- Graphs missing labels: 0
- Graphs missing nodes: 0
- Graphs missing edges: 0
- Graphs with invalid edges: 0
- Node/edge count errors: 0
- Label mismatches: 0
- Metadata missing graphs: 0
- Missing graph files: 0

Final status:

GRAPH DATASET VALIDATION PASSED

---

## 13. Handoff to GNN Component

The graph dataset produced by this pipeline can be provided to the
GNN component.

The main handoff directory is:

data/graphs/

The GNN component can use:

- graph structure from nodes and edges
- graph labels from the label field
- post text for transformer-based text embeddings
- timestamps and temporal features
- propagation structure for graph learning

The GNN stage is responsible for using these graph representations
for downstream model development and evaluation.

---

## 14. Person 1 Contribution

Person 1 was responsible for:

1. Dataset collection and organization
2. Post and claim preprocessing
3. Label extraction and normalization
4. Propagation relationship extraction
5. Propagation tree construction
6. Temporal feature extraction
7. Propagation feature extraction
8. Graph dataset generation
9. Dataset validation
10. Preparation of GNN-ready graph data

The final output of this contribution is a validated collection of
6,425 graph-structured PHEME conversation threads.
