# gnn_bet_selector.py
# Lightweight selector that approximates "GNN-Bet" behaviour using graph centralities
# Requires: numpy, networkx
# Save this single file into your project (no external model required).

import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class GNNBetSelector:
    def __init__(self, verbose: bool = True):
        """
        No external model. Use networkx centrality heuristics to pick important nodes.
        verbose: prints debug messages requested by you.
        """
        self.verbose = verbose

    def _normalize_edges(self, edges) -> np.ndarray:
        """
        Normalize various edge representations into a numpy array of shape (E,2), dtype=int64.
        Accepts:
          - None -> returns empty array shape (0,2)
          - numpy array shape (E,2) -> pass through
          - numpy array shape (2,E) -> transpose to (E,2)
          - list of pairs -> vstack
          - list of numpy arrays each shape (2,) -> vstack
          - torch tensor (2,E) or (E,2) -> convert to numpy
        """
        if edges is None:
            return np.zeros((0, 2), dtype=np.int64)

        # handle torch
        if TORCH_AVAILABLE and isinstance(edges, torch.Tensor):
            edges_np = edges.detach().cpu().numpy()
        else:
            edges_np = np.asarray(edges)

        # If it's a 1-D array of tuples / objects, attempt to vstack
        if edges_np.ndim == 1:
            # e.g., list of arrays or list of lists
            try:
                edges_np = np.vstack([np.asarray(e).reshape(2) for e in edges])
            except Exception:
                # fallback: try reshape
                try:
                    edges_np = edges_np.reshape(-1, 2)
                except Exception:
                    return np.zeros((0,2), dtype=np.int64)

        # Now edges_np is at least 2D
        if edges_np.ndim == 2:
            # if shape is (2, E) (common edge_index), transpose
            if edges_np.shape[0] == 2 and edges_np.shape[1] != 2:
                edges_np = edges_np.T
        else:
            # unexpected dims
            try:
                edges_np = edges_np.reshape(-1, 2)
            except Exception:
                return np.zeros((0,2), dtype=np.int64)

        # ensure dtype integer
        try:
            edges_np = edges_np.astype(np.int64)
        except Exception:
            edges_np = np.array(edges_np, dtype=np.int64)

        # if somehow shape[1] != 2, try to reshape
        if edges_np.ndim == 2 and edges_np.shape[1] != 2:
            try:
                edges_np = edges_np.reshape(-1, 2)
            except Exception:
                return np.zeros((0,2), dtype=np.int64)

        return edges_np

    def _score_with_centralities(self,
                                 node_features: np.ndarray,
                                 edges,
                                 method_weights: Dict[str, float],
                                 betweenness_k: Optional[int] = None) -> np.ndarray:
        """
        method_weights: dict like {'betweenness':0.5, 'pagerank':0.2, 'degree':0.1, 'featnorm':0.2}
        betweenness_k: if not None, approximate betweenness centrality with k node samples (speeds up big graphs)
        """
        if node_features is None:
            node_features = np.zeros((0, 0))
        N = int(node_features.shape[0]) if node_features.size != 0 else 0

        # normalize edges to shape (E,2)
        edges_np = self._normalize_edges(edges)

        if self.verbose:
            print(f"[GNN-BET-SELECTOR] edges normalized shape: {edges_np.shape}")

        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        if edges_np is not None and edges_np.shape[0] > 0:
            for u, v in edges_np:
                try:
                    G.add_edge(int(u), int(v))
                except Exception:
                    continue

        scores = np.zeros(N, dtype=float)

        if method_weights.get('betweenness', 0.0) > 0 and N > 0:
            try:
                bc = nx.betweenness_centrality(G, k=betweenness_k, normalized=True, weight=None, seed=42)
                arr = np.array([bc.get(i, 0.0) for i in range(N)], dtype=float)
            except Exception:
                arr = np.zeros(N, dtype=float)
            scores += method_weights.get('betweenness', 0.0) * arr

        if method_weights.get('pagerank', 0.0) > 0 and N > 0:
            try:
                pr = nx.pagerank(G, alpha=0.85)
                arr = np.array([pr.get(i, 0.0) for i in range(N)], dtype=float)
            except Exception:
                arr = np.zeros(N, dtype=float)
            scores += method_weights.get('pagerank', 0.0) * arr

        if method_weights.get('degree', 0.0) > 0 and N > 0:
            degs = np.array([G.out_degree(i) + G.in_degree(i) for i in range(N)], dtype=float)
            if degs.sum() > 0:
                degs = degs / degs.sum()
            scores += method_weights.get('degree', 0.0) * degs

        if method_weights.get('featnorm', 0.0) > 0 and N > 0:
            try:
                feat_norm = np.linalg.norm(node_features, axis=1)
                if feat_norm.sum() > 0:
                    feat_norm = feat_norm / feat_norm.sum()
                else:
                    feat_norm = np.ones(N, dtype=float) / float(N)
            except Exception:
                feat_norm = np.ones(N, dtype=float) / float(N)
            scores += method_weights.get('featnorm', 0.0) * feat_norm

        # normalize to [0,1]
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            if N > 0:
                scores = np.ones(N, dtype=float) / float(N)
            else:
                scores = np.zeros(0, dtype=float)
        return scores

    def graph_from_edges(self, edges, num_nodes: Optional[int] = None) -> nx.DiGraph:
        edges_np = self._normalize_edges(edges)
        if num_nodes is None:
            if edges_np is None or edges_np.shape[0] == 0:
                num_nodes = 0
            else:
                num_nodes = int(np.max(edges_np)) + 1
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        if edges_np is not None and edges_np.shape[0] > 0:
            for u, v in edges_np:
                try:
                    G.add_edge(int(u), int(v))
                except Exception:
                    continue
        return G

    def select_subgraph(self,
                        node_features: np.ndarray,
                        edges,
                        top_k_init: int = 1,
                        top_k_targets: int = 50,
                        max_nodes: Optional[int] = 2000,
                        coverage_ratio: Optional[float] = None,
                        method_weights: Optional[dict] = None,
                        betweenness_k: Optional[int] = None
                        ) -> Tuple[List[int], np.ndarray, Dict[int,int], np.ndarray]:
        """
        Returns:
          kept_global_nodes: sorted list of global node IDs (relative to the node_features & edges input)
          kept_edges_local: numpy array shape [E_kept, 2] of edges using local indices (0..len(kept)-1)
          local_map: dict global_id -> local_id
          node_feat_sub: numpy array shape [len(kept), D]
        """
        if method_weights is None:
            method_weights = {'betweenness': 0.5, 'pagerank': 0.2, 'degree': 0.1, 'featnorm': 0.2}

        N = int(node_features.shape[0]) if node_features is not None and node_features.size != 0 else 0
        if self.verbose:
            print(f"[GNN-BET-SELECTOR] Original graph nodes: {N}")

        # compute scores
        scores = self._score_with_centralities(node_features, edges, method_weights, betweenness_k=betweenness_k)

        # choose initial nodes
        if N == 0:
            return [], np.zeros((0,2), dtype=np.int64), {}, np.zeros((0, node_features.shape[1] if node_features.ndim>1 else 0), dtype=float)

        init_idx = np.argsort(-scores)[:top_k_init].tolist()
        if self.verbose:
            print(f"[GNN-BET-SELECTOR] initial node(s) chosen (global idx): {init_idx} (scores: {[scores[i] for i in init_idx]})")

        # build graph
        G = self.graph_from_edges(edges, num_nodes=N)

        # prepare targets (top_k_targets by score excluding init)
        sorted_idx = np.argsort(-scores)
        targets = [int(i) for i in sorted_idx if int(i) not in init_idx][:top_k_targets]

        kept = set(init_idx)

        # accumulate shortest-paths from init -> target
        for tgt in targets:
            best_path = None
            best_len = None
            for ini in init_idx:
                try:
                    path = nx.shortest_path(G, source=ini, target=tgt)
                    if best_len is None or len(path) < best_len:
                        best_len = len(path)
                        best_path = path
                except nx.NetworkXNoPath:
                    continue
            # try reverse path tgt->init if forward absent
            if best_path is None:
                for ini in init_idx:
                    try:
                        path = nx.shortest_path(G, source=tgt, target=ini)
                        if best_len is None or len(path) < best_len:
                            best_len = len(path)
                            best_path = path
                    except nx.NetworkXNoPath:
                        continue
            if best_path:
                kept.update(best_path)

            if max_nodes and len(kept) >= max_nodes:
                break
            if coverage_ratio and (len(kept) / float(N)) >= coverage_ratio:
                break

        kept = sorted(list(kept))
        if self.verbose:
            print(f"[GNN-BET-SELECTOR] kept nodes after shortest-path accumulation: {len(kept)} / {N}")

        # build local_map and kept_edges using local indices
        local_map = {g: i for i, g in enumerate(kept)}
        kept_edges = []
        edges_np = self._normalize_edges(edges)
        if edges_np is not None and edges_np.shape[0] > 0:
            for u, v in edges_np:
                gu = int(u); gv = int(v)
                if gu in local_map and gv in local_map:
                    kept_edges.append([local_map[gu], local_map[gv]])
        kept_edges = np.array(kept_edges, dtype=np.int64) if len(kept_edges) > 0 else np.zeros((0,2), dtype=np.int64)

        node_feat_sub = node_features[kept, :].copy() if N > 0 else np.zeros((0, node_features.shape[1] if node_features.ndim>1 else 0), dtype=float)

        print(f"[GNN-BET-DEBUG] original_nodes={N}, init_nodes={len(init_idx)}, kept_after_neighbors={len(kept)}, kept_edges={kept_edges.shape}")

        return kept, kept_edges, local_map, node_feat_sub
