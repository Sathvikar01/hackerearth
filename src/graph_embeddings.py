"""Graph Spatial Embeddings using Node2Vec.

Builds a geohash adjacency graph and learns dense spatial embeddings.
Uses BallTree for O(N log N) spatial lookups and Pearson-correlated
"behavioral" edges for traffic flow topology.
"""
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0


def build_geohash_graph(df: pd.DataFrame, method: str = "adjacency") -> nx.Graph:
    """Build a graph from geohash spatial relationships.

    Methods:
    - 'adjacency': Connect geohashes sharing geohash_prefix_4
    - 'distance': BallTree O(N log N) haversine edges within 2km
    - 'behavioral': Pearson-correlated demand patterns (Day 48 only)

    Args:
        df: DataFrame with 'geohash' column
        method: Graph construction method

    Returns:
        networkx Graph (edges have 'weight' attribute for behavioral)
    """
    G = nx.Graph()
    unique_geohashes = df["geohash"].unique()
    G.add_nodes_from(unique_geohashes)

    if method == "adjacency":
        prefix_groups = df.groupby("geohash_prefix_4")["geohash"].unique()
        for prefix, geohashes in prefix_groups.items():
            for i in range(len(geohashes)):
                for j in range(i + 1, len(geohashes)):
                    G.add_edge(geohashes[i], geohashes[j])

    elif method == "distance":
        # BallTree O(N log N) haversine lookup
        gh_coords = df.groupby("geohash")[["latitude", "longitude"]].mean()
        if len(gh_coords) == 0:
            return G

        # BallTree expects radians for haversine
        coords_rad = np.radians(gh_coords[["latitude", "longitude"]].values)
        tree = BallTree(coords_rad, metric="haversine")
        gh_list = list(gh_coords.index)

        # Query all pairs within 2km (radius in radians = km / earth_radius)
        radius = 2.0 / EARTH_RADIUS_KM
        indices = tree.query_radius(coords_rad, r=radius)

        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i < j:
                    G.add_edge(gh_list[i], gh_list[j])

    elif method == "behavioral":
        # Pearson-correlated demand patterns (Day 48 only)
        if "day" in df.columns:
            df_beh = df[df["day"] <= 48]
        else:
            df_beh = df

        # Pivot: rows=timestamp, cols=geohash, values=demand
        pivot = df_beh.pivot_table(
            index="timestamp", columns="geohash", values="demand", aggfunc="mean"
        ).fillna(0)

        if pivot.shape[1] > 1:
            corr_matrix = pivot.corr()
            threshold = 0.75
            for i, gh1 in enumerate(corr_matrix.columns):
                for j in range(i + 1, len(corr_matrix.columns)):
                    gh2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > threshold:
                        G.add_edge(gh1, gh2, weight=float(corr_val))

    return G


def compute_node2vec_embeddings(G: nx.Graph, dimensions: int = 16,
                                 walk_length: int = 20, num_walks: int = 50,
                                 p: float = 1.0, q: float = 1.0,
                                 workers: int = 1) -> pd.DataFrame:
    """Compute Node2Vec embeddings for all nodes in the graph."""
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, p=p, q=q, workers=workers,
                        quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {}
    for node in G.nodes():
        if str(node) in model.wv:
            embeddings[node] = model.wv[str(node)]
        elif node in model.wv:
            embeddings[node] = model.wv[node]

    emb_df = pd.DataFrame.from_dict(embeddings, orient="index")
    emb_df.columns = [f"n2v_{i}" for i in range(dimensions)]
    emb_df.index.name = "geohash"
    emb_df = emb_df.reset_index()
    return emb_df


def reduce_embedding_dimensions(emb_df: pd.DataFrame, n_components: int = 8,
                                 embedding_cols: list = None) -> pd.DataFrame:
    """Reduce embedding dimensions using PCA."""
    if embedding_cols is None:
        embedding_cols = [c for c in emb_df.columns if c.startswith("n2v_")]
    if len(embedding_cols) <= n_components:
        return emb_df

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(emb_df[embedding_cols].values)
    for i in range(n_components):
        emb_df[f"n2v_pca_{i}"] = reduced[:, i]

    keep_cols = ["geohash"] + [f"n2v_pca_{i}" for i in range(n_components)]
    return emb_df[keep_cols]


def add_graph_embeddings(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                          dimensions: int = 16, n_pca: int = 8,
                          method: str = "behavioral") -> tuple:
    """Full pipeline: build graph, compute Node2Vec, reduce, merge."""
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
    emb_df = reduce_embedding_dimensions(emb_df, n_components=n_pca)

    # Drop existing n2v columns to allow re-calling
    n2v_cols = [c for c in emb_df.columns if c.startswith("n2v_")]
    for col in n2v_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in val_or_test_df.columns:
            val_or_test_df = val_or_test_df.drop(columns=[col])

    train_df = train_df.merge(emb_df, on="geohash", how="left")
    val_or_test_df = val_or_test_df.merge(emb_df, on="geohash", how="left")

    for col in n2v_cols:
        train_df[col] = train_df[col].fillna(0.0)
        val_or_test_df[col] = val_or_test_df[col].fillna(0.0)

    print(f"    Added {n_pca} graph embedding features")
    return train_df, val_or_test_df
