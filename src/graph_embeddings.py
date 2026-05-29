"""Graph Spatial Embeddings using Node2Vec.

Builds a geohash adjacency graph and learns dense spatial embeddings.
Uses BallTree for O(N log N) spatial lookups and Pearson-correlated
"behavioral" edges for traffic flow topology.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree


def _try_import_networkx():
    """Try to import networkx, return None if not available."""
    try:
        import networkx as nx
        return nx
    except ImportError:
        return None


def _try_import_node2vec():
    """Try to import node2vec, return None if not available."""
    try:
        from node2vec import Node2Vec
        return Node2Vec
    except ImportError:
        return None


EARTH_RADIUS_KM = 6371.0


def build_geohash_graph(df: pd.DataFrame, method: str = "adjacency"):
    """Build a graph from geohash spatial relationships.

    Methods:
    - 'adjacency': Connect geohashes sharing geohash_prefix_4
    - 'distance': BallTree O(N log N) haversine edges within 2km
    - 'behavioral': Pearson-correlated demand patterns (Day 48 only)

    Args:
        df: DataFrame with 'geohash' column
        method: Graph construction method

    Returns:
        Dictionary-based graph: {"nodes": set, "edges": list of (u, v, weight)}
    """
    nx_module = _try_import_networkx()
    unique_geohashes = df["geohash"].unique().tolist()
    node_set = set(unique_geohashes)
    edges = []

    if method == "adjacency":
        if "geohash_prefix_4" not in df.columns:
            coords = df["geohash"].apply(lambda g: g[:4])
            df = df.copy()
            df["geohash_prefix_4"] = coords
        prefix_groups = df.groupby("geohash_prefix_4")["geohash"].unique()
        for prefix, geohashes in prefix_groups.items():
            geohashes = list(geohashes)
            for i in range(len(geohashes)):
                for j in range(i + 1, len(geohashes)):
                    edges.append((geohashes[i], geohashes[j], 1.0))

    elif method == "distance":
        if "latitude" not in df.columns or "longitude" not in df.columns:
            import pygeohash
            coords = df["geohash"].apply(lambda g: pygeohash.decode(g))
            lat = coords.apply(lambda x: x[0]).values
            lon = coords.apply(lambda x: x[1]).values
        else:
            lat = df["latitude"].values
            lon = df["longitude"].values

        gh_coords = {}
        for i, gh in enumerate(df["geohash"].values):
            if gh not in gh_coords:
                gh_coords[gh] = (lat[i], lon[i])

        gh_list = list(gh_coords.keys())
        coords_arr = np.radians([[gh_coords[g][0], gh_coords[g][1]] for g in gh_list])
        tree = BallTree(coords_arr, metric="haversine")
        radius = 2.0 / EARTH_RADIUS_KM
        indices = tree.query_radius(coords_arr, r=radius)

        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i < j:
                    edges.append((gh_list[i], gh_list[j], 1.0))

    elif method == "behavioral":
        if "day" in df.columns:
            df_beh = df[df["day"] <= 48]
        else:
            df_beh = df

        pivot = df_beh.pivot_table(
            index="timestamp", columns="geohash", values="demand", aggfunc="mean"
        ).fillna(0)

        if pivot.shape[1] > 1:
            corr_matrix = pivot.corr()
            threshold = 0.75
            cols = corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > threshold:
                        edges.append((cols[i], cols[j], float(corr_val)))

    return {"nodes": node_set, "edges": edges}


def compute_node2vec_embeddings(G: dict, dimensions: int = 16,
                                 walk_length: int = 20, num_walks: int = 50,
                                 p: float = 1.0, q: float = 1.0,
                                 workers: int = 1) -> pd.DataFrame:
    """Compute Node2Vec embeddings for all nodes in the graph."""
    Node2Vec = _try_import_node2vec()
    nx_module = _try_import_networkx()

    if Node2Vec is None or nx_module is None:
        print("    WARNING: node2vec or networkx not available, using random embeddings")
        nodes = list(G["nodes"])
        emb = {}
        for node in nodes:
            emb[node] = np.random.randn(dimensions)
        emb_df = pd.DataFrame.from_dict(emb, orient="index")
        emb_df.columns = [f"n2v_{i}" for i in range(dimensions)]
        emb_df.index.name = "geohash"
        emb_df = emb_df.reset_index()
        return emb_df

    G_nx = nx_module.Graph()
    G_nx.add_nodes_from(G["nodes"])
    for u, v, w in G["edges"]:
        G_nx.add_edge(u, v, weight=w)

    try:
        node2vec = Node2Vec(G_nx, dimensions=dimensions, walk_length=walk_length,
                            num_walks=num_walks, p=p, q=q, workers=workers,
                            quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        embeddings = {}
        for node in G_nx.nodes():
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]
            elif node in model.wv:
                embeddings[node] = model.wv[node]
            else:
                embeddings[node] = np.random.randn(dimensions)
    except Exception as e:
        print(f"    WARNING: Node2Vec failed ({e}), using random embeddings")
        nodes = list(G["nodes"])
        emb = {}
        for node in nodes:
            emb[node] = np.random.randn(dimensions)
        embeddings = emb

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
    n_nodes = len(G["nodes"])
    n_edges = len(G["edges"])
    print(f"    Graph: {n_nodes} nodes, {n_edges} edges")

    if n_edges == 0:
        print("    WARNING: No edges in graph, using zero embeddings")
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