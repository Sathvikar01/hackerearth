"""Graph Spatial Embeddings using Node2Vec.

Builds a geohash adjacency graph and learns dense spatial embeddings
that capture hidden spatial correlations (e.g., two business districts
far apart but with identical traffic patterns).
"""
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import PCA


def build_geohash_graph(df: pd.DataFrame, method: str = "adjacency") -> nx.Graph:
    """Build a graph from geohash spatial relationships.

    Methods:
    - 'adjacency': Connect geohashes that share a geohash_prefix_4 (neighborhood)
    - 'distance': Connect geohashes within a distance threshold
    - 'co-occurrence': Connect geohashes with correlated demand patterns

    Args:
        df: DataFrame with 'geohash' column
        method: Graph construction method

    Returns:
        networkx Graph
    """
    G = nx.Graph()

    unique_geohashes = df["geohash"].unique()
    G.add_nodes_from(unique_geohashes)

    if method == "adjacency":
        # Connect geohashes sharing the same prefix-4 (neighborhood)
        prefix_groups = df.groupby("geohash_prefix_4")["geohash"].unique()
        for prefix, geohashes in prefix_groups.items():
            for i in range(len(geohashes)):
                for j in range(i + 1, len(geohashes)):
                    G.add_edge(geohashes[i], geohashes[j])

    elif method == "distance":
        # Connect geohashes within lat/lon distance threshold
        if "latitude" in df.columns:
            gh_coords = df.groupby("geohash")[["latitude", "longitude"]].mean()
            threshold = 0.02  # ~2km
            for i, (gh1, coord1) in enumerate(gh_coords.iterrows()):
                for j, (gh2, coord2) in enumerate(gh_coords.iterrows()):
                    if i < j:
                        dist = np.sqrt(
                            (coord1["latitude"] - coord2["latitude"]) ** 2 +
                            (coord1["longitude"] - coord2["longitude"]) ** 2
                        )
                        if dist < threshold:
                            G.add_edge(gh1, gh2)

    return G


def compute_node2vec_embeddings(G: nx.Graph, dimensions: int = 16,
                                 walk_length: int = 20, num_walks: int = 50,
                                 p: float = 1.0, q: float = 1.0,
                                 workers: int = 1) -> pd.DataFrame:
    """Compute Node2Vec embeddings for all nodes in the graph.

    Args:
        G: networkx Graph
        dimensions: Embedding dimension
        walk_length: Length of each random walk
        num_walks: Number of walks per node
        p: Return parameter (controls likelihood of revisiting a node)
        q: In-out parameter (controls search local vs. global)
        workers: Number of parallel workers

    Returns:
        DataFrame with geohash index and embedding columns
    """
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, p=p, q=q, workers=workers,
                        quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings
    embeddings = {}
    for node in G.nodes():
        if str(node) in model.wv:
            embeddings[node] = model.wv[str(node)]
        elif node in model.wv:
            embeddings[node] = model.wv[node]

    # Create DataFrame
    emb_df = pd.DataFrame.from_dict(embeddings, orient="index")
    emb_df.columns = [f"n2v_{i}" for i in range(dimensions)]
    emb_df.index.name = "geohash"
    emb_df = emb_df.reset_index()

    return emb_df


def reduce_embedding_dimensions(emb_df: pd.DataFrame, n_components: int = 8,
                                 embedding_cols: list = None) -> pd.DataFrame:
    """Reduce embedding dimensions using PCA.

    Args:
        emb_df: DataFrame with embedding columns
        n_components: Number of PCA components
        embedding_cols: List of embedding column names (auto-detected if None)

    Returns:
        DataFrame with reduced embedding columns
    """
    if embedding_cols is None:
        embedding_cols = [c for c in emb_df.columns if c.startswith("n2v_")]

    if len(embedding_cols) <= n_components:
        return emb_df

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(emb_df[embedding_cols].values)

    for i in range(n_components):
        emb_df[f"n2v_pca_{i}"] = reduced[:, i]

    # Keep only PCA columns
    keep_cols = ["geohash"] + [f"n2v_pca_{i}" for i in range(n_components)]
    return emb_df[keep_cols]


def add_graph_embeddings(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                          dimensions: int = 16, n_pca: int = 8,
                          method: str = "adjacency") -> tuple:
    """Full pipeline: build graph, compute Node2Vec, reduce, merge.

    Args:
        train_df: Training DataFrame (used to build graph)
        val_or_test_df: Validation or test DataFrame
        dimensions: Node2Vec embedding dimensions
        n_pca: PCA components to keep
        method: Graph construction method

    Returns:
        (train_df, val_or_test_df) with embedding columns added
    """
    print("    Building geohash graph...")
    G = build_geohash_graph(train_df, method=method)
    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() == 0:
        print("    WARNING: No edges in graph, skipping embeddings")
        for i in range(n_pca):
            train_df[f"n2v_pca_{i}"] = 0.0
            val_or_test_df[f"n2v_pca_{i}"] = 0.0
        return train_df, val_or_test_df

    print("    Computing Node2Vec embeddings...")
    emb_df = compute_node2vec_embeddings(G, dimensions=dimensions)

    print("    Reducing dimensions with PCA...")
    emb_df = reduce_embedding_dimensions(emb_df, n_components=n_pca)

    # Drop existing n2v columns to allow re-calling
    n2v_cols = [c for c in emb_df.columns if c.startswith("n2v_")]
    for col in n2v_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in val_or_test_df.columns:
            val_or_test_df = val_or_test_df.drop(columns=[col])

    # Merge embeddings
    train_df = train_df.merge(emb_df, on="geohash", how="left")
    val_or_test_df = val_or_test_df.merge(emb_df, on="geohash", how="left")

    # Fill missing embeddings with 0
    for col in n2v_cols:
        train_df[col] = train_df[col].fillna(0.0)
        val_or_test_df[col] = val_or_test_df[col].fillna(0.0)

    print(f"    Added {n_pca} graph embedding features")
    return train_df, val_or_test_df
