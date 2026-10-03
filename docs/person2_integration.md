# Person 2 integration status

The Person 2 folder contains a validated frozen BERTweet pipeline using `vinai/bertweet-base`, attention-mask-aware mean pooling, 768-dimensional float32 vectors, and post-ID alignment. Its validation report states that 104,582 posts were embedded and that all graph nodes have exactly one aligned embedding.

The duplicate preprocessing, TF-IDF, tokenizer, and validation project should not be imported into the main model. The main repository now needs only the resulting `embeddings.npy` and `ids.csv` artifacts plus the small loader in `models/embedding_store.py`.

Expected artifact location:

```text
data/nlp/embeddings/full/embeddings.npy
data/nlp/embeddings/full/ids.csv
```

Once present, use `load_embedding_store(...)`, look up embeddings by each snapshot's `node_id` order, and pass the resulting matrix to `graph_from_json(..., node_embeddings=...)`. The graph model input dimension becomes `768 + 5 = 773`; GCN/GAT layers and the LOEO protocol remain unchanged.

The current checkout contains only Person 2 metadata/reports, not the large matrix or ID mapping. Therefore a text+graph benchmark has not been run yet.
